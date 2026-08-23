"""
cross_validation.py
====================
Single extra experiment: k-fold cross-validation (k=5) applied ONLY to the
best configuration found so far (phased, base config, 224x224), to verify
whether the holdout split's result (0.661 accuracy) was representative or
just a lucky/unlucky partition.

Design choice: the TEST set stays FIXED (the same 15% used in every other
experiment of this work), so the final result remains comparable. K-fold
is applied only over the remaining 85% (train+val), splitting it into 5
folds; in each fold, 4/5 are used for training and 1/5 for validation.

Run:
    python -m evaluation.cross_validation

Outputs (in outputs/):
    cv_results.csv       per-fold metrics
    cv_summary.txt        mean +/- std across folds
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score
from tqdm import tqdm

from data.lumos_dataset import LumosDataset
from torch.utils.data import DataLoader
from models.resnet import LumosNet, get_device
from training.strategies.phased import PhasedStrategy
from training.train import set_seed, compute_loss, train_one_epoch, evaluate as evaluate_epoch

ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "res" / "processed"
OUT_DIR = ROOT / "outputs"

N_FOLDS = 5
SEED = 42
EPOCHS = 50
PATIENCE = 7
BATCH_SIZE = 32


def main():
    OUT_DIR.mkdir(exist_ok=True)
    device = get_device()

    images = np.load(PROC_DIR / "images.npy")
    meta = pd.read_csv(PROC_DIR / "metadata.csv")

    # Fixed test set: identical to every other experiment in this work.
    test_mask = (meta["split"] == "test").values
    trainval_mask = ~test_mask

    trainval_meta = meta[trainval_mask].reset_index(drop=True)
    trainval_images = images[trainval_mask]

    # one row per patient (each patient's images share the same label)
    patients = trainval_meta.groupby("patient_id")["label"].first().reset_index()

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_results = []

    for fold, (train_pat_idx, val_pat_idx) in enumerate(
        skf.split(patients["patient_id"], patients["label"]), start=1
    ):
        print(f"\n{'='*60}\nFOLD {fold}/{N_FOLDS}\n{'='*60}")
        set_seed(SEED)  # same seed every fold: only the data partition changes

        train_patients = set(patients.iloc[train_pat_idx]["patient_id"])
        val_patients = set(patients.iloc[val_pat_idx]["patient_id"])

        train_mask = trainval_meta["patient_id"].isin(train_patients).values
        val_mask = trainval_meta["patient_id"].isin(val_patients).values

        train_ds = LumosDataset(trainval_images[train_mask], trainval_meta[train_mask], train=True)
        val_ds = LumosDataset(trainval_images[val_mask], trainval_meta[val_mask], train=False)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        print(f"train: {len(train_ds)} images | val: {len(val_ds)} images")

        model = LumosNet(pretrained=True).to(device)
        strategy = PhasedStrategy()  # base config: lr_finetune=1e-4, weight_decay=0
        optimizer = strategy.build_optimizer(model)

        ce = nn.CrossEntropyLoss()
        mse = nn.MSELoss()

        best_val, best_epoch, no_improve = float("inf"), 0, 0
        best_state = None

        for epoch in range(1, EPOCHS + 1):
            new_opt = strategy.on_epoch_start(model, epoch)
            if new_opt is not None:
                optimizer = new_opt

            train_loss = train_one_epoch(model, train_loader, optimizer, ce, mse, device)
            val_loss, val_acc, val_rmse = evaluate_epoch(model, val_loader, ce, mse, device)

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

        # ---- evaluate this fold's best model on the FIXED test set, per patient ----
        test_meta = meta[test_mask].reset_index(drop=True)
        test_images = images[test_mask]
        test_ds = LumosDataset(test_images, test_meta, train=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        model.eval()
        all_probs, all_bmd_pred = [], []
        with torch.no_grad():
            for imgs, _, _ in test_loader:
                imgs = imgs.to(device)
                logits, bmd_pred = model(imgs)
                all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
                all_bmd_pred.append(bmd_pred.cpu().numpy())
        probs = np.concatenate(all_probs)
        bmd_pred = np.concatenate(all_bmd_pred)

        df = test_meta[["patient_id", "label", "bmd"]].copy()
        df[["p0", "p1", "p2"]] = probs
        df["bmd_pred"] = bmd_pred
        agg = df.groupby("patient_id").agg(
            label=("label", "first"), bmd_true=("bmd", "first"),
            p0=("p0", "mean"), p1=("p1", "mean"), p2=("p2", "mean"),
            bmd_pred=("bmd_pred", "mean"),
        ).reset_index()
        agg["pred"] = agg[["p0", "p1", "p2"]].values.argmax(axis=1)

        acc = (agg["pred"] == agg["label"]).mean()
        rep = classification_report(agg["label"], agg["pred"], output_dict=True, zero_division=0)
        f1_macro = rep["macro avg"]["f1-score"]
        try:
            auc = roc_auc_score(agg["label"], agg[["p0", "p1", "p2"]].values, multi_class="ovr")
        except ValueError:
            auc = float("nan")
        rmse = np.sqrt(np.mean((agg["bmd_true"] - agg["bmd_pred"]) ** 2))
        pcc = np.corrcoef(agg["bmd_true"], agg["bmd_pred"])[0, 1]

        print(f"  FOLD {fold} TEST -> acc {acc:.3f} | f1 {f1_macro:.3f} | "
              f"auc {auc:.3f} | rmse {rmse:.4f} | pcc {pcc:.3f}")

        fold_results.append({
            "fold": fold, "best_epoch": best_epoch, "best_val_loss": best_val,
            "test_acc": acc, "test_f1_macro": f1_macro, "test_auc": auc,
            "test_rmse": rmse, "test_pcc": pcc,
        })

    # ---- summary across folds ----
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(OUT_DIR / "cv_results.csv", index=False)

    summary_lines = ["Cross-validation summary (k=5, phased, base config, fixed test set)\n"]
    for col in ["test_acc", "test_f1_macro", "test_auc", "test_rmse", "test_pcc"]:
        mean, std = results_df[col].mean(), results_df[col].std()
        summary_lines.append(f"{col:15s}: {mean:.4f} +/- {std:.4f}")
    summary_text = "\n".join(summary_lines)

    print("\n" + "=" * 60)
    print(summary_text)
    (OUT_DIR / "cv_summary.txt").write_text(summary_text)
    print(f"\nSaved: {OUT_DIR / 'cv_results.csv'} and cv_summary.txt")


if __name__ == "__main__":
    main()