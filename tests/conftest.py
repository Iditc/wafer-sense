"""Shared fixtures: small synthetic wafers, so tests need no real data or GPU."""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest
import torch

import src.models.dataset as dataset_module

MAP_SIZE = 64


def circle_mask(size: int = MAP_SIZE) -> np.ndarray:
    """Boolean mask of a round wafer inside a square map."""
    center = (size - 1) / 2
    y, x = np.ogrid[:size, :size]
    return (x - center) ** 2 + (y - center) ** 2 <= (size / 2) ** 2


def make_maps(n: int, seed: int = 0) -> np.ndarray:
    """Random 0/1/2 wafer maps: 0 outside, 1 good die, 2 defective die."""
    rng = np.random.default_rng(seed)
    mask = circle_mask()
    maps = np.zeros((n, MAP_SIZE, MAP_SIZE), dtype=np.uint8)
    for wafer in maps:
        wafer[mask] = 1 + (rng.random(int(mask.sum())) < 0.1)
    return maps


@pytest.fixture
def small_map() -> np.ndarray:
    """A 4x4 asymmetric map with all three values."""
    return np.array(
        [[0, 1, 1, 0], [1, 2, 1, 1], [1, 1, 2, 2], [0, 1, 1, 0]], dtype=np.uint8
    )


@pytest.fixture
def wafer_batch() -> torch.Tensor:
    """Batch of 8 encoded wafers: wafer-area and defect channels."""
    maps = torch.from_numpy(make_maps(8))
    return torch.stack([maps > 0, maps == 2], dim=1).float()


@pytest.fixture
def tiny_cache(tmp_path, monkeypatch) -> dict[str, np.ndarray]:
    """Fake memory-mapped cache with train and val rows, all 9 classes in each."""
    train_y = np.repeat(np.arange(9), 10)
    val_y = np.repeat(np.arange(9), 2)
    labels = {"train": train_y, "val": val_y}
    rows = {
        "train": np.arange(len(train_y)),
        "val": np.arange(len(train_y), len(train_y) + len(val_y)),
    }
    maps_path = tmp_path / "maps.npy"
    np.save(maps_path, make_maps(len(train_y) + len(val_y)))
    monkeypatch.setattr(
        dataset_module, "cache_paths", lambda size, subset: (maps_path, None)
    )
    monkeypatch.setattr(
        dataset_module,
        "split_index",
        lambda split, size=64: (rows[split].copy(), labels[split].copy()),
    )
    return labels
