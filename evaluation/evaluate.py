"""
evaluate.py
===========
Evaluates a trained LumosNet on the TEST set, aggregating predictions
PER PATIENT (images of each patient combined by averaging softmax
probabilities; BMD is the average of their predicted values).

Supports both the full dataset and the AP+ROI dataset via --data.

Run:
    python -m evaluation.evaluate --strategy differential --data full
    python -m evaluation.evaluate --strategy phased        --data roi

Outputs (in outputs/):
    confusion_<strategy>_<data>.png
    bmd_scatter_<strategy>_<data>.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

from data.dataloaders import make_dataloaders, PROC_DIR
from models.lumosnet import LumosNet, get_device

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"
CLASS_NAMES = ["Sano", "Osteopenia", "Osteoporosis"]

DATASETS = {
    "full": {"img_file": "images.npy", "meta_file": "metadata.csv"},
    "roi":  {"img_file": "images_roi.npy", "meta_file": "metadata_roi.csv"},
    "roi_hybrid": {"img_file": "images_roi_hybrid.npy", "meta_file": "metadata_roi_hybrid.csv"},
    "hires": {"img_file": "images_320.npy", "meta_file": "metadata_320.csv"},
}


# =====================================================================
# INFERENCE ON TEST (per image)
# =====================================================================
@torch.no_grad()
def predict_test(model, loader, device):
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
def aggregate_by_patient(probs, bmd_pred, meta_file):
    meta = pd.read_csv(PROC_DIR / meta_file)
    test = meta[meta["split"] == "test"].reset_index(drop=True)

    assert len(test) == len(probs), (
        f"Misaligned: {len(test)} test rows vs {len(probs)} predictions. "
        "Make sure the test loader is not shuffled."
    )

    df = test[["patient_id", "label", "bmd"]].copy()
    df[["p0", "p1", "p2"]] = probs
    df["bmd_pred"] = bmd_pred

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


# =====================================================================
# MAIN
# =====================================================================
def evaluate(strategy: str, data_name: str):
    OUT_DIR.mkdir(exist_ok=True)
    device = get_device()

    files = DATASETS[data_name]
    tag = f"{strategy}_{data_name}"

    ckpt = OUT_DIR / f"best_{tag}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"No trained model found: {ckpt}. Train first.")

    loaders = make_dataloaders(batch_size=32, **files)
    model = LumosNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"Loaded {ckpt} | Device: {device}")

    probs, bmd_pred = predict_test(model, loaders["test"], device)
    agg = aggregate_by_patient(probs, bmd_pred, files["meta_file"])
    print(f"\nTest patients: {len(agg)}  (aggregated from {len(probs)} images)")

    cm = classification_metrics(agg)
    regression_metrics(agg)

    plot_confusion(cm, tag)
    plot_bmd_scatter(agg, tag)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["differential", "phased"], default="phased")
    parser.add_argument("--data", choices=list(DATASETS.keys()), default="full")
    args = parser.parse_args()
    evaluate(args.strategy, args.data)