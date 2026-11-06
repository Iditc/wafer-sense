"""Train the autoencoder on O.K. wafers and evaluate it as a defect detector on val.

No defect labels are used for training or for choosing the best epoch (lowest
rebuild loss on O.K. val wafers). Labels are used only to measure the result.
Run (Colab GPU): python -m src.models.train_ae --name ae_code128
Quick local check: python -m src.models.train_ae --limit 2000 --epochs 2 --workers 0 --name _smoke
"""
import argparse
import time

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import CHECKPOINT_DIR, MAX_MISSED_DEFECT_RATE, RANDOM_SEED
from src.data.plot_utils import save, setup_style
from src.models import ae_eval
from src.models.autoencoder import WaferAE
from src.models.cnn import count_params
from src.models.dataset import WaferDataset
from src.models.metrics import OK_INDEX, save_metrics
from src.models.train_utils import get_device, set_seed


def parse_args() -> argparse.Namespace:
    """Command-line settings; the defaults are the 128-number code run."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="ae_code128")
    p.add_argument("--code-dim", type=int, default=128, help="size of the bottleneck code")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--false-alarm", type=float, default=0.014,
                   help="O.K. share to flag when comparing with the CNN (its val rate)")
    p.add_argument("--limit", type=int, default=None, help="use only N wafers per split (testing)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    return p.parse_args()


def only_ok(dataset: WaferDataset) -> WaferDataset:
    """Keep the O.K. wafers of a dataset (in place) and return it."""
    keep = dataset.labels == OK_INDEX
    dataset.rows, dataset.labels = dataset.rows[keep], dataset.labels[keep]
    return dataset


def make_loaders(args: argparse.Namespace) -> dict[str, DataLoader]:
    """O.K. train (augmented), O.K. val (epoch choice) and all val wafers (scoring)."""
    kwargs = {"batch_size": args.batch_size, "num_workers": args.workers,
              "pin_memory": torch.cuda.is_available(), "persistent_workers": args.workers > 0}
    train = only_ok(WaferDataset("train", train_aug=True, limit=args.limit, seed=args.seed))
    val_ok = only_ok(WaferDataset("val", limit=args.limit))
    val_all = WaferDataset("val", limit=args.limit)
    return {"train": DataLoader(train, shuffle=True, **kwargs),
            "val_ok": DataLoader(val_ok, shuffle=False, **kwargs),
            "val_all": DataLoader(val_all, shuffle=False, **kwargs)}


def run_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device,
              optimizer: torch.optim.Optimizer | None = None) -> float:
    """Mean rebuild loss over a loader; trains when an optimizer is given."""
    model.train(optimizer is not None)
    total, n = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            loss = loss_fn(model(x), x)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(x)
            n += len(x)
    return total / n


def fit(model: nn.Module, loaders: dict, args: argparse.Namespace, device: torch.device) -> list[dict]:
    """Train with cosine lr; keep the epoch with the lowest O.K. val loss."""
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CHECKPOINT_DIR / f"{args.name}.pt"
    best, bad, history = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        start, lr = time.time(), optimizer.param_groups[0]["lr"]
        train_loss = run_epoch(model, loaders["train"], loss_fn, device, optimizer)
        scheduler.step()
        val_loss = run_epoch(model, loaders["val_ok"], loss_fn, device)
        history.append({"epoch": epoch, "lr": lr, "train_loss": train_loss, "val_ok_loss": val_loss})
        print(f"epoch {epoch:2d} | lr {lr:.2e} | train {train_loss:.5f} | val O.K. {val_loss:.5f} | "
              f"{time.time() - start:.0f}s")
        if val_loss < best:
            best, bad = val_loss, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stop: no val gain for {args.patience} epochs")
                break
    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"Best val O.K. loss {best:.5f}, weights in {ckpt.name}")
    return history


def plot_history(history: list[dict], name: str) -> None:
    """Learning rate, train loss and O.K. val loss per epoch."""
    setup_style()
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    for ax, key, title, color in zip(axes, ["lr", "train_loss", "val_ok_loss"],
                                     ["Learning rate", "Train loss", "Val O.K. loss"],
                                     ["gray", "tab:blue", "darkorange"]):
        ax.plot(epochs, [h[key] for h in history], marker="o", color=color)
        ax.set(xlabel="Epoch", ylabel=title, title=title)
    save(fig, f"{name}_history.png")


def main() -> None:
    """Train, score every val wafer, and save metrics and figures."""
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    loaders = make_loaders(args)
    model = WaferAE(code_dim=args.code_dim).to(device)
    print(f"{count_params(model):,} parameters | device {device} | "
          f"{len(loaders['train'].dataset):,} O.K. train / {len(loaders['val_all'].dataset):,} val wafers")

    history = fit(model, loaders, args, device)
    y, scores = ae_eval.compute_scores(model, loaders["val_all"], device)
    result = ae_eval.evaluate_scores(y, scores, args.false_alarm, MAX_MISSED_DEFECT_RATE)
    print("Val:", ae_eval.summary_line(result))
    save_metrics({"split": "val", "config": vars(args), "history": history, **result}, args.name)

    best, op = result["best_score"], result["at_false_alarm"]
    ae_eval.plot_binary_confusion(
        op["confusion_matrix"], f"AE ({args.name}) on val, {op['false_alarm_rate']:.1%} false alarms",
        f"{args.name}_val_confusion.png")
    ae_eval.plot_scores_by_class(
        scores[best], y, op["threshold"], f"AE ({args.name}): {best} score per class on val",
        f"{args.name}_scores.png")
    ae_eval.plot_examples(model, loaders["val_all"].dataset, scores[best],
                          f"AE ({args.name}): typical val wafers", f"{args.name}_examples.png", device)
    plot_history(history, args.name)


if __name__ == "__main__":
    main()
