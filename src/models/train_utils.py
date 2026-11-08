"""Training loop shared by the supervised classifiers.

Early stopping uses F1 macro on val, never the test set. The best weights
are saved to checkpoints/<name>.pt and reloaded at the end of training.
"""
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import CHECKPOINT_DIR, RANDOM_SEED
from src.data.plot_utils import CLASSES
from src.models.dataset import WaferDataset, sample_labels
from src.models.metrics import evaluate, summary_line


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Make runs repeatable across Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    """GPU when available (Colab), otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loaders(size: int = 64, batch_size: int = 256, num_workers: int = 2,
                 limit: int | None = None, channels: int = 2, seed: int = RANDOM_SEED,
                 label_frac: float | None = None, exclude: str | None = None) -> dict[str, DataLoader]:
    """Train (augmented, shuffled) and val (as is) loaders.

    label_frac: train on this share of each class's train labels (the same wafers
    the linear probe gets). exclude: drop one defect type from train and val, so the
    model never sees it (fair baseline for the new-type test).
    """
    train = WaferDataset("train", size, train_aug=True, limit=limit, seed=seed, channels=channels)
    val = WaferDataset("val", size, limit=limit, channels=channels)
    if label_frac is not None:
        train.keep(sample_labels(train.labels, label_frac))
    if exclude is not None:
        left_out = CLASSES.index(exclude)
        for data in (train, val):
            data.keep(np.flatnonzero(data.labels != left_out))
    kwargs = {"batch_size": batch_size, "num_workers": num_workers,
              "pin_memory": torch.cuda.is_available(), "persistent_workers": num_workers > 0}
    return {"train": DataLoader(train, shuffle=True, **kwargs),
            "val": DataLoader(val, shuffle=False, **kwargs)}


def train_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module,
                optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    """One pass over the training data; returns the mean loss."""
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / n


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """True labels and predicted class probabilities for a whole loader."""
    model.eval()
    ys, probs = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device)), dim=1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(probs).numpy()


def fit(model: nn.Module, loaders: dict[str, DataLoader], loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer, name: str, epochs: int = 30, patience: int = 8,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None) -> list[dict]:
    """Train with early stopping on val F1 macro; returns the per-epoch history.

    If a scheduler is given, it is stepped once per epoch (e.g. cosine decay).
    """
    device = get_device()
    model.to(device)
    loss_fn.to(device)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CHECKPOINT_DIR / f"{name}.pt"
    best_f1, bad_epochs, history = -1.0, 0, []

    for epoch in range(1, epochs + 1):
        start = time.time()
        lr = optimizer.param_groups[0]["lr"]
        loss = train_epoch(model, loaders["train"], loss_fn, optimizer, device)
        if scheduler is not None:
            scheduler.step()
        y, probs = predict(model, loaders["val"], device)
        metrics = evaluate(y, probs.argmax(axis=1))
        history.append({"epoch": epoch, "lr": lr, "train_loss": loss, "val_f1_macro": metrics["f1_macro"]})
        print(f"epoch {epoch:2d} | lr {lr:.2e} | loss {loss:.4f} | val {summary_line(metrics)} | "
              f"{time.time() - start:.0f}s")

        if metrics["f1_macro"] > best_f1:
            best_f1, bad_epochs = metrics["f1_macro"], 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stop: no val gain for {patience} epochs")
                break

    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"Best val F1 macro {best_f1:.3f}, weights in {ckpt.name}")
    return history
