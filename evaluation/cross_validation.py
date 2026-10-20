import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score
from torch.utils.data import DataLoader

from data.df_import import PROC_DIR
from data.lumos_dataset import LumosDataset
from evaluation.evaluate import PROB_COLS, aggregate_by_patient, predict_test
from models.lumosnet import LumosNet, get_device
from training.strategies.phased import PhasedStrategy
from training.train import (
    BATCH_SIZE, EPOCHS, OUT_DIR, PATIENCE, SEED, evaluate as evaluate_epoch, set_seed, train_one_epoch,
)

N_FOLDS = 5


# Entrena un fold con early stopping y devuelve el modelo con los mejores pesos,
# la mejor época y la mejor pérdida de validación
def train_fold(train_ds, val_ds, device):
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = LumosNet(pretrained=True).to(device)
    strategy = PhasedStrategy()
    optimizer = strategy.build_optimizer(model)
    ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()

    best_val, best_epoch, no_improve, best_state = float("inf"), 0, 0, None

    for epoch in range(1, EPOCHS + 1):
        new_opt = strategy.on_epoch_start(model, epoch)
        if new_opt is not None:
            optimizer = new_opt

        train_loss = train_one_epoch(model, train_loader, optimizer, ce, mse, device, use_covariates=False)
        val_loss, val_acc, val_rmse = evaluate_epoch(model, val_loader, ce, mse, device, use_covariates=False)

        if val_loss < best_val:
            best_val, best_epoch, no_improve = val_loss, epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        print(f"  epoch {epoch:2d} | train {train_loss:.4f} | val {val_loss:.4f} "
              f"| acc {val_acc:.3f} | rmse {val_rmse:.4f}"
              + ("  <- best" if no_improve == 0 else f"  ({no_improve}/{PATIENCE})"))

        if no_improve >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} (best: {best_epoch})")
            break

    model.load_state_dict(best_state)
    return model, best_epoch, best_val


def test_metrics(agg: pd.DataFrame) -> dict:
    rep = classification_report(agg["label"], agg["pred"], output_dict=True, zero_division=0)
    try:
        auc = roc_auc_score(agg["label"], agg[PROB_COLS].values, multi_class="ovr")
    except ValueError:
        auc = float("nan")
    return {
        "test_acc": (agg["pred"] == agg["label"]).mean(),
        "test_f1_macro": rep["macro avg"]["f1-score"],
        "test_auc": auc,
        "test_rmse": np.sqrt(np.mean((agg["bmd_true"] - agg["bmd_pred"]) ** 2)),
        "test_pcc": np.corrcoef(agg["bmd_true"], agg["bmd_pred"])[0, 1],
    }


def main():
    OUT_DIR.mkdir(exist_ok=True)
    device = get_device()

    images = np.load(PROC_DIR / "images.npy")
    meta = pd.read_csv(PROC_DIR / "metadata.csv")

    test_mask = (meta["split"] == "test").values
    trainval_meta = meta[~test_mask].reset_index(drop=True)
    trainval_images = images[~test_mask]

    test_ds = LumosDataset(images[test_mask], meta[test_mask], train=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Folds por paciente, no por imagen
    patients = trainval_meta.groupby("patient_id")["label"].first().reset_index()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(patients["patient_id"], patients["label"]), start=1):
        print(f"\n{'=' * 60}\nFOLD {fold}/{N_FOLDS}\n{'=' * 60}")
        set_seed(SEED)

        train_mask = trainval_meta["patient_id"].isin(patients.iloc[train_idx]["patient_id"]).values
        val_mask = trainval_meta["patient_id"].isin(patients.iloc[val_idx]["patient_id"]).values
        train_ds = LumosDataset(trainval_images[train_mask], trainval_meta[train_mask], train=True)
        val_ds = LumosDataset(trainval_images[val_mask], trainval_meta[val_mask], train=False)
        print(f"train: {len(train_ds)} images | val: {len(val_ds)} images")

        model, best_epoch, best_val = train_fold(train_ds, val_ds, device)

        probs, bmd_pred = predict_test(model, test_loader, device, use_covariates=False)
        metrics = test_metrics(aggregate_by_patient(test_ds.meta, probs, bmd_pred))

        print(f"  FOLD {fold} TEST -> acc {metrics['test_acc']:.3f} | f1 {metrics['test_f1_macro']:.3f} | "
              f"auc {metrics['test_auc']:.3f} | rmse {metrics['test_rmse']:.4f} | pcc {metrics['test_pcc']:.3f}")
        fold_results.append({"fold": fold, "best_epoch": best_epoch, "best_val_loss": best_val, **metrics})

    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(OUT_DIR / "cv_results.csv", index=False)

    summary_lines = [f"Cross-validation summary (k={N_FOLDS}, phased, base config, fixed test set)\n"]
    for col in ["test_acc", "test_f1_macro", "test_auc", "test_rmse", "test_pcc"]:
        summary_lines.append(f"{col:15s}: {results_df[col].mean():.4f} +/- {results_df[col].std():.4f}")
    summary_text = "\n".join(summary_lines)

    print("\n" + "=" * 60)
    print(summary_text)
    (OUT_DIR / "cv_summary.txt").write_text(summary_text)
    print(f"\nSaved: {OUT_DIR / 'cv_results.csv'} and cv_summary.txt")


if __name__ == "__main__":
    main()
