"""Train the baseline CNN and evaluate it on val (test stays closed).

Run (Colab GPU): python -m src.models.train_cnn
Quick local check: python -m src.models.train_cnn --limit 2000 --epochs 2 --workers 0 --name _smoke
"""
import argparse

import matplotlib.pyplot as plt
import torch
from torch import nn

from src.data.plot_utils import save, setup_style
from src.models.cnn import WaferCNN, count_params
from src.models.dataset import class_weights
from src.models.metrics import evaluate, plot_confusion, save_metrics, summary_line
from src.models.train_utils import fit, get_device, make_loaders, predict, set_seed


def parse_args() -> argparse.Namespace:
    """Command-line settings; the defaults are the baseline run."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="cnn_baseline", help="name for metrics, figures and weights")
    p.add_argument("--size", type=int, default=64, help="input map size (cache must exist)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--no-weights", action="store_true", help="plain loss, no class weights")
    p.add_argument("--limit", type=int, default=None, help="use only N wafers per split (testing)")
    p.add_argument("--workers", type=int, default=2)
    return p.parse_args()


def plot_history(history: list[dict], name: str) -> None:
    """Train loss and val F1 macro per epoch, side by side."""
    setup_style()
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(epochs, [h["train_loss"] for h in history], marker="o")
    ax1.set(xlabel="Epoch", ylabel="Train loss", title="Train loss")
    ax2.plot(epochs, [h["val_f1_macro"] for h in history], marker="o", color="darkorange")
    ax2.set(xlabel="Epoch", ylabel="Val F1 macro", title="Val F1 macro")
    save(fig, f"{name}_history.png")


def main() -> None:
    """Train, pick the best epoch on val, and save val metrics and figures."""
    args = parse_args()
    set_seed()
    loaders = make_loaders(args.size, args.batch_size, args.workers, args.limit)
    model = WaferCNN()
    print(f"{count_params(model):,} parameters | device {get_device()} | "
          f"{len(loaders['train'].dataset):,} train / {len(loaders['val'].dataset):,} val wafers")

    weight = None if args.no_weights else class_weights(loaders["train"].dataset.labels)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = fit(model, loaders, loss_fn, optimizer, args.name, args.epochs, args.patience)

    y, probs = predict(model, loaders["val"], get_device())
    metrics = evaluate(y, probs.argmax(axis=1))
    print("Val:", summary_line(metrics))
    save_metrics({"split": "val", "config": vars(args), "history": history, **metrics}, args.name)
    plot_confusion(metrics["confusion_matrix"],
                   f"Baseline CNN on val: F1 macro {metrics['f1_macro']:.3f}",
                   f"{args.name}_val_confusion.png")
    plot_history(history, args.name)


if __name__ == "__main__":
    main()
