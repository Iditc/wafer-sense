"""Extra embeddings of the frozen model for the day 3b similarity comparison.

Two variants, both for the labeled train and val wafers, in the order of the
frozen embeddings:
  rot_mean - the final 256-number embedding averaged over the 8 rotations and
             flips of the wafer, so a wafer and its rotated copy match exactly
  block3   - the 128-channel output of the third conv block, average pooled:
             an earlier, less class-specific view of the wafer
The plain (identity) embedding is recomputed too and must match the frozen one.
Run: python -m src.features.view_embeddings  (about an hour on an 8-core CPU)
"""

import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import PROCESSED_DIR
from src.features.frozen import (
    check_embeddings,
    load_frozen_embeddings,
    load_frozen_model,
)
from src.models.cnn import WaferCNN
from src.models.dataset import WaferDataset
from src.models.train_utils import get_device

VIEWS_FILE = PROCESSED_DIR / "emb_cnn_cosine_views.npz"
SPLITS = ("train", "val")


def symmetries(x: torch.Tensor) -> list[torch.Tensor]:
    """The 8 rotations and flips of a batch of maps (N x C x H x W)."""
    turns = [torch.rot90(x, k, dims=(2, 3)) for k in range(4)]
    return turns + [t.flip(3) for t in turns]


@torch.no_grad()
def view_embeddings(
    model: WaferCNN, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Plain embedding, mean embedding over the 8 symmetries, and pooled block 3."""
    block3 = model.blocks[:3](x)
    plain = model.pool(model.blocks[3](block3)).flatten(1)
    rot_sum = plain.clone()
    for view in symmetries(x)[1:]:
        rot_sum += model.embed(view)
    return plain, rot_sum / 8, model.pool(block3).flatten(1)


def embed_split(
    model: WaferCNN, split: str, device: torch.device, batch: int, workers: int
) -> dict[str, np.ndarray]:
    """All three embeddings and labels of one split's labeled wafers."""
    loader = DataLoader(WaferDataset(split), batch_size=batch, num_workers=workers)
    parts: dict[str, list] = {"plain": [], "rot_mean": [], "block3": [], "y": []}
    start = time.time()
    for i, (x, y) in enumerate(loader):
        for name, value in zip(
            ("plain", "rot_mean", "block3"), view_embeddings(model, x.to(device))
        ):
            parts[name].append(value.cpu())
        parts["y"].append(y)
        if i % 20 == 0:
            done = (i + 1) * batch
            print(f"{split}: {done:,} wafers, {time.time() - start:.0f} s", flush=True)
    return {name: torch.cat(values).numpy() for name, values in parts.items()}


def main() -> None:
    """Embed train and val, check the plain embeddings, save one npz file."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--workers", type=int, default=0)
    args = p.parse_args()

    torch.set_num_threads(os.cpu_count() or 1)
    device = get_device()
    model = load_frozen_model(device)
    out = {}
    for split in SPLITS:
        result = embed_split(model, split, device, args.batch, args.workers)
        frozen_x, frozen_y = load_frozen_embeddings(split)
        if not np.array_equal(result["y"], frozen_y):
            raise ValueError(f"{split}: wafer order differs from the frozen embeddings")
        lowest = check_embeddings(result["plain"], frozen_x)
        print(
            f"{split}: plain embedding matches the frozen one (lowest cosine {lowest:.6f})"
        )
        out[f"{split}_rot_mean"] = result["rot_mean"]
        out[f"{split}_block3"] = result["block3"]
        out[f"{split}_y"] = result["y"]
    np.savez(VIEWS_FILE, **out)
    print(f"Saved {VIEWS_FILE}")


if __name__ == "__main__":
    main()
