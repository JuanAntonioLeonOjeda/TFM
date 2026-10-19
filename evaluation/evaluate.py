"""Evalúa un checkpoint en test, agregando por paciente (media de probabilidades y de BMD).

Uso: python -m evaluation.evaluate --strategy phased --data full [--class-weighting inverse] [--covariates]
     (los flags deben coincidir con los usados en training.train)
"""

import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

from data.dataloaders import DATASETS, MEDICAL_COVARIATES, make_dataloaders
from data.df_import import CLASS_NAMES
from models.lumosnet import LumosNet, get_device
from training.train import OUT_DIR, add_experiment_args, run_tag

PROB_COLS = ["p0", "p1", "p2"]


@torch.no_grad()
def predict_test(model, loader, device, use_covariates: bool):
    model.eval()
    all_probs, all_bmd = [], []
    for batch in loader:
        images = batch[0].to(device)
        covariates = batch[3].to(device) if use_covariates else None
        logits, bmd_pred = model(images, covariates)
        all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        all_bmd.append(bmd_pred.cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_bmd)


def aggregate_by_patient(test_meta: pd.DataFrame, probs, bmd_pred) -> pd.DataFrame:
    """test_meta debe estar en el mismo orden que las predicciones (loader sin shuffle)."""
    assert len(test_meta) == len(probs), (
        f"Misaligned: {len(test_meta)} test rows vs {len(probs)} predictions."
    )
    df = test_meta[["patient_id", "label", "bmd"]].copy()
    df[PROB_COLS] = probs
    df["bmd_pred"] = bmd_pred

    agg = df.groupby("patient_id").agg(
        label=("label", "first"),
        bmd_true=("bmd", "first"),
        p0=("p0", "mean"), p1=("p1", "mean"), p2=("p2", "mean"),
        bmd_pred=("bmd_pred", "mean"),
    ).reset_index()
    agg["pred"] = agg[PROB_COLS].values.argmax(axis=1)
    return agg


def classification_metrics(agg):
    y_true = agg["label"].values
    y_pred = agg["pred"].values

    print("\n--- CLASSIFICATION (per patient) ---")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3))

    cm = confusion_matrix(y_true, y_pred)

    print("Per-class sensitivity / specificity:")
    total = cm.sum()
    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        sens = tp / (tp + fn) if (tp + fn) else 0
        spec = tn / (tn + fp) if (tn + fp) else 0
        print(f"  {name:13s}: sens {sens:.3f} | spec {spec:.3f}")

    try:
        auc = roc_auc_score(y_true, agg[PROB_COLS].values, multi_class="ovr")
        print(f"\nAUC (macro, one-vs-rest): {auc:.3f}")
    except ValueError as e:
        print(f"\nAUC not computed: {e}")

    return cm


def regression_metrics(agg):
    y_true = agg["bmd_true"].values
    y_pred = agg["bmd_pred"].values

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    pcc = np.corrcoef(y_true, y_pred)[0, 1]

    print("\n--- REGRESSION: BMD (per patient) ---")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE : {mae:.4f}")
    print(f"  PCC : {pcc:.4f}")
    return rmse, mae, pcc


def plot_confusion(cm, tag):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(3)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, fraction=0.046)
    plt.tight_layout()
    path = OUT_DIR / f"confusion_{tag}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    print(f"\nSaved: {path}")


def plot_bmd_scatter(agg, tag):
    y_true, y_pred = agg["bmd_true"].values, agg["bmd_pred"].values
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors="none")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Real"); ax.set_ylabel("Prediction")
    plt.tight_layout()
    path = OUT_DIR / f"bmd_scatter_{tag}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    print(f"Saved: {path}")


def evaluate(strategy: str, data_name: str, class_weighting: str = "none",
             use_covariates: bool = False):
    OUT_DIR.mkdir(exist_ok=True)
    device = get_device()

    tag = run_tag(strategy, data_name, class_weighting, use_covariates)
    ckpt = OUT_DIR / f"best_{tag}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"No trained model found: {ckpt}. Entrena primero con training.train "
                                f"usando los mismos flags.")

    covariate_cols = MEDICAL_COVARIATES if use_covariates else None
    loaders = make_dataloaders(batch_size=32, covariate_cols=covariate_cols, **DATASETS[data_name])

    model = LumosNet(pretrained=False, num_covariates=len(covariate_cols or [])).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"Loaded {ckpt} | Device: {device}")

    probs, bmd_pred = predict_test(model, loaders["test"], device, use_covariates)
    agg = aggregate_by_patient(loaders["test"].dataset.meta, probs, bmd_pred)
    print(f"\nTest patients: {len(agg)}  (aggregated from {len(probs)} images)")

    cm = classification_metrics(agg)
    regression_metrics(agg)

    plot_confusion(cm, tag)
    plot_bmd_scatter(agg, tag)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_experiment_args(parser)
    args = parser.parse_args()
    evaluate(args.strategy, args.data, args.class_weighting, args.covariates)
