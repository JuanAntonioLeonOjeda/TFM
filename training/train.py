"""
train.py
========
Generic training loop for the multitask LumosNet (classification + BMD regression).
Supports both the full dataset and the AP+ROI dataset via --data.
The transfer-learning strategy is pluggable (see training/strategies/).

Run:
    python -m training.train --strategy differential --data full
    python -m training.train --strategy phased        --data roi

Outputs (in outputs/):
    best_<strategy>_<data>.pt      best model weights
    history_<strategy>_<data>.png  loss / accuracy / rmse curves
"""

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from data.dataloaders import make_dataloaders
from models.lumosnet import LumosNet, get_device
from training.strategies.differential import DifferentialStrategy
from training.strategies.phased import PhasedStrategy

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"

EPOCHS = 50
BATCH_SIZE = 32
BMD_WEIGHT = 10.0
PATIENCE = 7
SEED = 42

STRATEGIES = {
    "differential": DifferentialStrategy,
    "phased": PhasedStrategy,
}

# which files each --data option points to
DATASETS = {
    "full": {"img_file": "images.npy", "meta_file": "metadata.csv"},
    "roi":  {"img_file": "images_roi.npy", "meta_file": "metadata_roi.csv"},
    "roi_hybrid": {"img_file": "images_roi_hybrid.npy", "meta_file": "metadata_roi_hybrid.csv"},
    "hires": {"img_file": "images_320.npy", "meta_file": "metadata_320.csv"},
}

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =====================================================================
# LOSS
# =====================================================================
def compute_loss(logits, bmd_pred, labels, bmd_true, ce, mse):
    loss_cls = ce(logits, labels)
    loss_reg = mse(bmd_pred, bmd_true)
    return loss_cls + BMD_WEIGHT * loss_reg


# =====================================================================
# TRAIN / EVAL FOR ONE EPOCH
# =====================================================================
def train_one_epoch(model, loader, optimizer, ce, mse, device):
    model.train()
    running = 0.0
    for images, labels, bmd in tqdm(loader, leave=False, desc="train"):
        images, labels, bmd = images.to(device), labels.to(device), bmd.to(device)

        optimizer.zero_grad()
        logits, bmd_pred = model(images)
        loss = compute_loss(logits, bmd_pred, labels, bmd, ce, mse)
        loss.backward()
        optimizer.step()

        running += loss.item() * images.size(0)
    return running / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, ce, mse, device):
    model.eval()
    running, correct, sq_err, n = 0.0, 0, 0.0, 0
    for images, labels, bmd in loader:
        images, labels, bmd = images.to(device), labels.to(device), bmd.to(device)

        logits, bmd_pred = model(images)
        loss = compute_loss(logits, bmd_pred, labels, bmd, ce, mse)

        running += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        sq_err += ((bmd_pred - bmd) ** 2).sum().item()
        n += images.size(0)

    return running / n, correct / n, (sq_err / n) ** 0.5


# =====================================================================
# MAIN TRAINING LOOP
# =====================================================================
def train(strategy_name: str, data_name: str):
    set_seed(SEED)
    OUT_DIR.mkdir(exist_ok=True)
    device = get_device()

    strategy = STRATEGIES[strategy_name]()
    files = DATASETS[data_name]
    tag = f"{strategy_name}_{data_name}"
    print(f"Strategy: {strategy_name} | Data: {data_name} | Device: {device} | Patience: {PATIENCE}\n")

    loaders = make_dataloaders(batch_size=BATCH_SIZE, **files)
    model = LumosNet(pretrained=True).to(device)

    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    optimizer = strategy.build_optimizer(model)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_rmse": []}
    best_val = float("inf")
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        new_opt = strategy.on_epoch_start(model, epoch)
        if new_opt is not None:
            optimizer = new_opt
            print(f"--- Strategy switched optimizer at epoch {epoch} ---")

        train_loss = train_one_epoch(model, loaders["train"], optimizer, ce, mse, device)
        val_loss, val_acc, val_rmse = evaluate(model, loaders["val"], ce, mse, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_rmse"].append(val_rmse)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), OUT_DIR / f"best_{tag}.pt")
            flag = "  <- best"
        else:
            epochs_no_improve += 1
            flag = f"  (no improve {epochs_no_improve}/{PATIENCE})"

        print(f"Epoch {epoch:2d}/{EPOCHS} | train {train_loss:.4f} | "
              f"val {val_loss:.4f} | acc {val_acc:.3f} | rmse {val_rmse:.4f}{flag}")

        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}: no improvement in {PATIENCE} epochs.")
            break

    print(f"\nBest val loss: {best_val:.4f} (epoch {best_epoch})")
    print(f"Saved: {OUT_DIR / f'best_{tag}.pt'}")

    plot_history(history, tag, best_epoch)
    return history


def plot_history(history, tag, best_epoch):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label="best")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()

    axes[1].plot(epochs, history["val_acc"], color="green")
    axes[1].axvline(best_epoch, color="gray", linestyle="--", linewidth=1)
    axes[1].set_title("Val accuracy (classification)"); axes[1].set_xlabel("epoch")

    axes[2].plot(epochs, history["val_rmse"], color="red")
    axes[2].axvline(best_epoch, color="gray", linestyle="--", linewidth=1)
    axes[2].set_title("Val RMSE (BMD regression)"); axes[2].set_xlabel("epoch")

    plt.suptitle(f"{tag}  (best epoch: {best_epoch})")
    plt.tight_layout()
    path = OUT_DIR / f"history_{tag}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    print(f"Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=list(STRATEGIES.keys()), default="phased")
    parser.add_argument("--data", choices=list(DATASETS.keys()), default="full",
                        help="'full' = both views, 'roi' = AP-only cropped")
    args = parser.parse_args()

    train(args.strategy, args.data)