import argparse
from pathlib import Path

import torch
import torch.nn as nn
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

# ---- hyperparameters ----
EPOCHS = 30
BATCH_SIZE = 32
BMD_WEIGHT = 10.0          # weight of the regression loss vs classification loss

# registry: name -> strategy class
STRATEGIES = {
    "differential": DifferentialStrategy,
    "phased": PhasedStrategy,
}


# =====================================================================
# LOSS
# =====================================================================
def compute_loss(logits, bmd_pred, labels, bmd_true, ce, mse):
    """Combined multitask loss: classification + weighted regression."""
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

    return running / n, correct / n, (sq_err / n) ** 0.5   # loss, accuracy, rmse


# =====================================================================
# MAIN TRAINING LOOP
# =====================================================================
def train(strategy_name: str):
    OUT_DIR.mkdir(exist_ok=True)
    device = get_device()

    strategy = STRATEGIES[strategy_name]()
    print(f"Strategy: {strategy_name} | Device: {device}\n")

    loaders = make_dataloaders(batch_size=BATCH_SIZE)
    model = LumosNet(pretrained=True).to(device)

    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    optimizer = strategy.build_optimizer(model)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_rmse": []}
    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # let the strategy change things at the start of the epoch (e.g. unfreeze).
        # if it returns a new optimizer, we switch to it.
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

        print(f"Epoch {epoch:2d}/{EPOCHS} | train {train_loss:.4f} | "
              f"val {val_loss:.4f} | acc {val_acc:.3f} | rmse {val_rmse:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), OUT_DIR / f"best_{strategy_name}.pt")

    print(f"\nBest val loss: {best_val:.4f}")
    print(f"Saved: {OUT_DIR / f'best_{strategy_name}.pt'}")

    plot_history(history, strategy_name)
    return history


def plot_history(history, strategy_name):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()

    axes[1].plot(epochs, history["val_acc"], color="green")
    axes[1].set_title("Val accuracy (classification)"); axes[1].set_xlabel("epoch")

    axes[2].plot(epochs, history["val_rmse"], color="red")
    axes[2].set_title("Val RMSE (BMD regression)"); axes[2].set_xlabel("epoch")

    plt.suptitle(f"Strategy: {strategy_name}")
    plt.tight_layout()
    path = OUT_DIR / f"history_{strategy_name}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    print(f"Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=list(STRATEGIES.keys()),
                        default="differential", help="transfer-learning strategy")
    args = parser.parse_args()

    train(args.strategy)