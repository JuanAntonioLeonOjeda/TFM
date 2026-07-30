"""
evaluate.py
===========
Evaluates a trained LumosNet on the TEST set, aggregating predictions
PER PATIENT (as decided): the images of each patient are combined into a
single diagnosis by averaging their softmax probabilities, and the BMD is
the average of their predicted values.

Run (after training):
    python -m evaluation.evaluate --strategy differential
    python -m evaluation.evaluate --strategy phased

Outputs (in outputs/):
    confusion_<strategy>.png     confusion matrix (per patient)
    bmd_scatter_<strategy>.png   predicted vs true BMD (per patient)
    and a full report printed to console.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
)

from data.dataloaders import make_dataloaders, PROC_DIR
from models.resnet import LumosNet, get_device

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"
CLASS_NAMES = ["Sano", "Osteopenia", "Osteoporosis"]


# =====================================================================
# INFERENCE ON TEST (per image)
# =====================================================================
@torch.no_grad()
def predict_test(model, loader, device):
    """Returns per-image softmax probs (N,3) and predicted BMD (N,)."""
    model.eval()
    all_probs, all_bmd = [], []
    for images, _, _ in loader:
        images = images.to(device)
        logits, bmd_pred = model(images)
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_bmd.append(bmd_pred.cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_bmd)


# =====================================================================
# PER-PATIENT AGGREGATION
# =====================================================================
def aggregate_by_patient(probs, bmd_pred):
    """
    Combines the per-image predictions into one prediction per patient.
    Uses the TEST rows of metadata.csv (same order as the test loader:
    shuffle=False, drop_last=False).
    """
    meta = pd.read_csv(PROC_DIR / "metadata.csv")
    test = meta[meta["split"] == "test"].reset_index(drop=True)

    assert len(test) == len(probs), (
        f"Misaligned: {len(test)} test rows vs {len(probs)} predictions. "
        "Make sure the test loader is not shuffled."
    )

    df = test[["patient_id", "label", "bmd"]].copy()
    df[["p0", "p1", "p2"]] = probs
    df["bmd_pred"] = bmd_pred

    # one row per patient: mean of probs and bmd_pred, true values are constant
    agg = df.groupby("patient_id").agg(
        label=("label", "first"),
        bmd_true=("bmd", "first"),
        p0=("p0", "mean"), p1=("p1", "mean"), p2=("p2", "mean"),
        bmd_pred=("bmd_pred", "mean"),
    ).reset_index()

    agg["pred"] = agg[["p0", "p1", "p2"]].values.argmax(axis=1)
    return agg


# =====================================================================
# METRICS
# =====================================================================
def classification_metrics(agg):
    y_true = agg["label"].values
    y_pred = agg["pred"].values
    probs = agg[["p0", "p1", "p2"]].values

    print("\n--- CLASSIFICATION (per patient) ---")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3))

    cm = confusion_matrix(y_true, y_pred)

    # per-class sensitivity (recall) and specificity from the confusion matrix
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

    # multiclass AUC (one-vs-rest)
    try:
        auc = roc_auc_score(y_true, probs, multi_class="ovr")
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


# =====================================================================
# PLOTS
# =====================================================================
def plot_confusion(cm, strategy):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(3)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión — {strategy}")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, fraction=0.046)
    plt.tight_layout()
    path = OUT_DIR / f"confusion_{strategy}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    print(f"\nSaved: {path}")


def plot_bmd_scatter(agg, strategy):
    y_true, y_pred = agg["bmd_true"].values, agg["bmd_pred"].values
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors="none")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1)   # perfect prediction line
    ax.set_xlabel("BMD real"); ax.set_ylabel("BMD predicho")
    ax.set_title(f"BMD predicho vs real — {strategy}")
    plt.tight_layout()
    path = OUT_DIR / f"bmd_scatter_{strategy}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    print(f"Saved: {path}")


# =====================================================================
# MAIN
# =====================================================================
def evaluate(strategy: str):
    OUT_DIR.mkdir(exist_ok=True)
    device = get_device()

    ckpt = OUT_DIR / f"best_{strategy}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"No trained model found: {ckpt}. Train first.")

    loaders = make_dataloaders(batch_size=32)
    model = LumosNet(pretrained=False).to(device)   # weights come from the checkpoint
    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"Loaded {ckpt} | Device: {device}")

    probs, bmd_pred = predict_test(model, loaders["test"], device)
    agg = aggregate_by_patient(probs, bmd_pred)
    print(f"\nTest patients: {len(agg)}  (aggregated from {len(probs)} images)")

    cm = classification_metrics(agg)
    regression_metrics(agg)

    plot_confusion(cm, strategy)
    plot_bmd_scatter(agg, strategy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["differential", "phased"],
                        default="differential")
    args = parser.parse_args()
    evaluate(args.strategy)