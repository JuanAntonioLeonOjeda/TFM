import argparse
import json
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from data.dataloaders import DATASETS, MEDICAL_COVARIATES, load_meta, make_dataloaders
from data.df_import import PROJECT_ROOT
from data.lumos_dataset import compute_covariate_stats
from models.lumosnet import LumosNet, get_device
from training.strategies.differential import DifferentialStrategy
from training.strategies.phased import PhasedStrategy

# Carpeta de checkpoints y figuras. La API la lee de MODEL_DIR para poder montarla como volumen
OUT_DIR = Path(os.environ.get("MODEL_DIR", PROJECT_ROOT / "outputs"))

EPOCHS = 50
BATCH_SIZE = 32
BMD_WEIGHT = 10.0
PATIENCE = 7
SEED = 42
NUM_CLASSES = 3

STRATEGIES = {
    "differential": DifferentialStrategy,
    "phased": PhasedStrategy,
}
CLASS_WEIGHTING = ["none", "inverse", "effective"]


# Nombre del experimento, usado en los checkpoints y en las figuras
def run_tag(strategy: str, data_name: str, class_weighting: str = "none",
            use_covariates: bool = False) -> str:
    tag = f"{strategy}_{data_name}"
    if class_weighting != "none":
        tag += f"_cw{class_weighting}"
    if use_covariates:
        tag += "_covars"
    return tag


def covariate_stats_path(tag: str):
    return OUT_DIR / f"best_{tag}_covariates.json"


# Guarda junto al checkpoint la media y la desviación de train de cada covariable, y la codificación
# de las columnas de texto, para poder normalizar pacientes nuevos sin necesidad del dataset
def save_covariate_stats(data_name: str, tag: str):
    meta_file = DATASETS[data_name]["meta_file"]
    raw = load_meta(meta_file)
    meta = load_meta(meta_file, MEDICAL_COVARIATES)
    stats = compute_covariate_stats(meta[meta["split"] == "train"], MEDICAL_COVARIATES)
    codes = {col: {str(v).lower(): i for i, v in enumerate(sorted(raw[col].dropna().unique()))}
             for col in MEDICAL_COVARIATES if not pd.api.types.is_numeric_dtype(raw[col])}
    spec = {
        "covariates": MEDICAL_COVARIATES,
        "mean": {c: float(stats[c][0]) for c in MEDICAL_COVARIATES},
        "std": {c: float(stats[c][1]) for c in MEDICAL_COVARIATES},
        "codes": codes,
    }
    OUT_DIR.mkdir(exist_ok=True)
    path = covariate_stats_path(tag)
    path.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {path}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(labels: torch.Tensor, num_classes: int = NUM_CLASSES,
                          scheme: str = "none", beta: float = 0.999):
    if scheme == "none":
        return None

    counts = torch.bincount(labels, minlength=num_classes).float()
    if scheme == "inverse":
        weights = counts.sum() / (num_classes * counts)
    elif scheme == "effective":
        # Cui et al. 2019, "Class-Balanced Loss Based on Effective Number of Samples"
        effective_num = 1.0 - torch.pow(torch.tensor(beta), counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * num_classes
    else:
        raise ValueError(f"Esquema de balanceo desconocido: {scheme}")
    return weights


def compute_loss(logits, bmd_pred, labels, bmd_true, ce, mse):
    return ce(logits, labels) + BMD_WEIGHT * mse(bmd_pred, bmd_true)


# El batch tiene imagen, etiqueta y BMD, y además las covariables si se usan
def unpack_batch(batch, use_covariates: bool, device):
    images, labels, bmd = (t.to(device) for t in batch[:3])
    covariates = batch[3].to(device) if use_covariates else None
    return images, labels, bmd, covariates


def train_one_epoch(model, loader, optimizer, ce, mse, device, use_covariates: bool):
    model.train()
    running = 0.0
    for batch in tqdm(loader, leave=False, desc="train"):
        images, labels, bmd, covariates = unpack_batch(batch, use_covariates, device)

        optimizer.zero_grad()
        logits, bmd_pred = model(images, covariates)
        loss = compute_loss(logits, bmd_pred, labels, bmd, ce, mse)
        loss.backward()
        optimizer.step()

        running += loss.item() * images.size(0)
    return running / len(loader.dataset)


# Devuelve la pérdida, la exactitud y el RMSE calculados por imagen
@torch.no_grad()
def evaluate(model, loader, ce, mse, device, use_covariates: bool):
    model.eval()
    running, correct, sq_err, n = 0.0, 0, 0.0, 0
    for batch in loader:
        images, labels, bmd, covariates = unpack_batch(batch, use_covariates, device)

        logits, bmd_pred = model(images, covariates)
        loss = compute_loss(logits, bmd_pred, labels, bmd, ce, mse)

        running += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        sq_err += ((bmd_pred - bmd) ** 2).sum().item()
        n += images.size(0)

    return running / n, correct / n, (sq_err / n) ** 0.5


def train(strategy_name: str, data_name: str, class_weighting: str = "none",
          use_covariates: bool = False):
    set_seed(SEED)
    OUT_DIR.mkdir(exist_ok=True)
    device = get_device()

    strategy = STRATEGIES[strategy_name]()
    tag = run_tag(strategy_name, data_name, class_weighting, use_covariates)
    covariate_cols = MEDICAL_COVARIATES if use_covariates else None

    print(f"Strategy: {strategy_name} | Data: {data_name} | Device: {device} | "
          f"Patience: {PATIENCE} | Class weighting: {class_weighting} | "
          f"Covariates: {covariate_cols or 'none'}\n")

    loaders = make_dataloaders(batch_size=BATCH_SIZE, covariate_cols=covariate_cols, **DATASETS[data_name])
    model = LumosNet(pretrained=True, num_covariates=len(covariate_cols or [])).to(device)
    if use_covariates:
        save_covariate_stats(data_name, tag)

    train_labels = torch.as_tensor(loaders["train"].dataset.meta["label"].values, dtype=torch.long)
    class_weights = compute_class_weights(train_labels, scheme=class_weighting)
    if class_weights is not None:
        print(f"Pesos de clase ({class_weighting}): {[round(w, 3) for w in class_weights.tolist()]}\n")
        class_weights = class_weights.to(device)
    ce = nn.CrossEntropyLoss(weight=class_weights)
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

        train_loss = train_one_epoch(model, loaders["train"], optimizer, ce, mse, device, use_covariates)
        val_loss, val_acc, val_rmse = evaluate(model, loaders["val"], ce, mse, device, use_covariates)

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


# Argumentos que identifican un experimento, compartidos por train y evaluate
def add_experiment_args(parser: argparse.ArgumentParser):
    parser.add_argument("--strategy", choices=list(STRATEGIES), default="phased")
    parser.add_argument("--data", choices=list(DATASETS), default="full")
    parser.add_argument("--class-weighting", choices=CLASS_WEIGHTING, default="none")
    parser.add_argument("--covariates", action="store_true", help=f"Añade {MEDICAL_COVARIATES}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_experiment_args(parser)
    parser.add_argument("--export-stats", action="store_true",
                        help="Solo genera el JSON de covariables de un modelo ya entrenado, sin entrenar")
    args = parser.parse_args()
    if args.export_stats:
        save_covariate_stats(args.data, run_tag(args.strategy, args.data, args.class_weighting, True))
        raise SystemExit
    train(args.strategy, args.data, args.class_weighting, args.covariates)
