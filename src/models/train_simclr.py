"""Train SimCLR on train-split wafers, labeled and unlabeled, with no labels at all.

Each epoch samples --samples wafers. The full training state is saved after every
epoch to --state-dir, and a rerun with the same --name resumes from it, so a Colab
disconnect loses at most one epoch. The embeddings are evaluated separately (day 6).
Run (Colab GPU): python -m src.models.train_simclr --name simclr_t02 --temperature 0.2
Quick local check: python -m src.models.train_simclr --subset labeled --limit 2000
                   --samples 1024 --batch-size 128 --epochs 2 --workers 0 --name _smoke
"""
import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, RandomSampler

from src.config import CHECKPOINT_DIR, RANDOM_SEED
from src.data.plot_utils import save, setup_style
from src.models.cnn import count_params
from src.models.dataset import MapDataset
from src.models.metrics import save_metrics
from src.models.simclr import ADD_MAX, DROP_MAX, SimCLR, nt_xent, random_view
from src.models.train_utils import get_device, set_seed


def parse_args() -> argparse.Namespace:
    """Command-line settings; the defaults are the main run (temperature 0.2)."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="simclr_t02")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--add-max", type=float, default=ADD_MAX, help="die noise: max share of dies added (0 = off)")
    p.add_argument("--drop-max", type=float, default=DROP_MAX, help="die noise: max share of defects dropped (0 = off)")
    p.add_argument("--samples", type=int, default=200_000, help="wafers sampled per epoch")
    p.add_argument("--subset", default="all", choices=["all", "labeled"],
                   help="map cache to read: all wafers, or labeled only (for quick tests)")
    p.add_argument("--limit", type=int, default=None, help="use only N train wafers (testing)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--state-dir", type=Path, default=CHECKPOINT_DIR,
                   help="where the per-epoch training state is saved (Drive on Colab)")
    return p.parse_args()


def make_loader(args: argparse.Namespace) -> DataLoader:
    """Train wafers in random order, --samples of them per epoch, full batches only."""
    data = MapDataset("train", subset=args.subset, limit=args.limit, seed=args.seed)
    sampler = RandomSampler(data, num_samples=min(args.samples, len(data)))
    return DataLoader(data, batch_size=args.batch_size, sampler=sampler, drop_last=True,
                      num_workers=args.workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=args.workers > 0)


def train_epoch(model: SimCLR, loader: DataLoader, optimizer: torch.optim.Optimizer,
                scaler: torch.amp.GradScaler, args: argparse.Namespace, device: torch.device) -> tuple[float, float]:
    """One pass; returns the mean loss and the mean pair accuracy."""
    model.train()
    use_amp = device.type == "cuda"
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x in loader:
        x = x.to(device, non_blocking=True)
        views = torch.cat([random_view(x, args.add_max, args.drop_max) for _ in range(2)])  # one batch for BatchNorm
        with torch.autocast(device.type, enabled=use_amp):
            z = model(views)
        loss, acc = nt_xent(*z.chunk(2), args.temperature)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss, total_acc, n = total_loss + loss.item(), total_acc + acc, n + 1
    return total_loss / n, total_acc / n


def fit(model: SimCLR, loader: DataLoader, args: argparse.Namespace, device: torch.device) -> list[dict]:
    """Train for a fixed number of epochs with cosine lr, resuming if a saved state exists."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    args.state_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.state_dir / f"{args.name}_state.pt"
    history, start_epoch = [], 1
    if state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=False)
        for obj, key in ((model, "model"), (optimizer, "optimizer"), (scheduler, "scheduler"), (scaler, "scaler")):
            obj.load_state_dict(state[key])
        history, start_epoch = state["history"], state["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        start, lr = time.time(), optimizer.param_groups[0]["lr"]
        loss, acc = train_epoch(model, loader, optimizer, scaler, args, device)
        scheduler.step()
        history.append({"epoch": epoch, "lr": lr, "loss": loss, "pair_acc": acc})
        print(f"epoch {epoch:2d} | lr {lr:.2e} | loss {loss:.4f} | pair acc {acc:.1%} | "
              f"{time.time() - start:.0f}s")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "history": history}, state_path)
    return history


def plot_history(history: list[dict], name: str) -> None:
    """Learning rate, loss and pair accuracy per epoch."""
    setup_style()
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    for ax, key, title, color in zip(axes, ["lr", "loss", "pair_acc"],
                                     ["Learning rate", "NT-Xent loss", "Pair accuracy"],
                                     ["gray", "tab:blue", "darkorange"]):
        ax.plot(epochs, [h[key] for h in history], marker="o", color=color)
        ax.set(xlabel="Epoch", ylabel=title, title=title)
    save(fig, f"{name}_history.png")


def plot_views(data: MapDataset, args: argparse.Namespace, n_wafers: int = 4, n_views: int = 4,
               seed: int = 0) -> None:
    """A few train wafers next to random views of them, to check the augmentation."""
    setup_style()
    gen = torch.Generator().manual_seed(seed)
    picks = torch.randint(len(data), (n_wafers,), generator=gen).tolist()
    x = torch.stack([data[i] for i in picks])
    fig, axes = plt.subplots(n_wafers, n_views + 1, figsize=(3 * (n_views + 1), 3 * n_wafers))
    for col in range(n_views + 1):
        batch = x if col == 0 else random_view(x, args.add_max, args.drop_max)
        for row in range(n_wafers):
            ax = axes[row, col]
            ax.imshow((batch[row, 0] + batch[row, 1]).numpy(), cmap="viridis", vmin=0, vmax=2,
                      interpolation="nearest")
            ax.axis("off")
            if row == 0:
                ax.set_title("Original" if col == 0 else f"View {col}")
    save(fig, f"{args.name}_views.png")


def main() -> None:
    """Train, save the final weights, history and figures."""
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    loader = make_loader(args)
    model = SimCLR().to(device)
    print(f"{count_params(model):,} parameters | device {device} | {len(loader.dataset):,} train wafers, "
          f"{len(loader) * args.batch_size:,} per epoch | temperature {args.temperature}")
    plot_views(loader.dataset, args)

    history = fit(model, loader, args, device)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_DIR / f"{args.name}.pt")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    save_metrics({"config": config, "history": history}, args.name)
    plot_history(history, args.name)
    print(f"Final loss {history[-1]['loss']:.4f}, pair acc {history[-1]['pair_acc']:.1%}, "
          f"weights in {args.name}.pt")


if __name__ == "__main__":
    main()
