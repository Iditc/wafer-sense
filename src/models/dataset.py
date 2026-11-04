"""PyTorch dataset over the cached fixed-size wafer maps.

Map values 0/1/2 are categories, not quantities, so each map is one-hot
encoded into 2 binary channels: channel 0 = wafer area (any die),
channel 1 = defective dies. Labels are integers 0..8 in the order of CLASSES.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import SPLIT_PARQUET, WAFERS_PARQUET
from src.data.cache_maps import cache_paths
from src.data.plot_utils import CLASSES

CLASS_TO_INDEX = {cls: i for i, cls in enumerate(CLASSES)}


def split_index(split: str, size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Rows of the labeled cache that belong to one split, and their integer labels."""
    _, ids_path = cache_paths(size, "labeled")
    cache = pd.DataFrame({"wafer_id": np.load(ids_path)})
    cache["row"] = np.arange(len(cache))
    meta = pd.read_parquet(SPLIT_PARQUET, columns=["wafer_id", "split"]).merge(
        pd.read_parquet(WAFERS_PARQUET, columns=["wafer_id", "failure_type"]), on="wafer_id"
    )
    rows = cache.merge(meta, on="wafer_id", how="left")
    rows = rows[rows["split"] == split]
    labels = rows["failure_type"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    return rows["row"].to_numpy(), labels


def encode(wafer_map: np.ndarray, channels: int = 2) -> np.ndarray:
    """Turn a 0/1/2 map into model input.

    channels=2: one-hot, wafer area and defects (the default).
    channels=1: the raw values scaled to 0 / 0.5 / 1 (used only for comparison).
    """
    if channels == 1:
        return (wafer_map[None] / 2).astype(np.float32)
    return np.stack([wafer_map > 0, wafer_map == 2]).astype(np.float32)


def augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random 90-degree rotation and horizontal flip (8 variants, label unchanged)."""
    x = np.rot90(x, k=int(rng.integers(4)), axes=(1, 2))
    if rng.random() < 0.5:
        x = x[:, :, ::-1]
    return np.ascontiguousarray(x)


def class_weights(labels: np.ndarray) -> torch.Tensor:
    """Loss weights proportional to 1/sqrt(class count), scaled to mean 1."""
    counts = np.bincount(labels, minlength=len(CLASSES)).astype(np.float64)
    weights = 1.0 / np.sqrt(np.maximum(counts, 1))
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


class WaferDataset(Dataset):
    """Labeled wafers of one split, read lazily from the memory-mapped cache."""

    def __init__(self, split: str, size: int = 64, train_aug: bool = False,
                 limit: int | None = None, seed: int = 0, channels: int = 2):
        self.maps_path, _ = cache_paths(size, "labeled")
        self.channels = channels
        self.rows, self.labels = split_index(split, size)
        if limit is not None:
            pick = np.random.default_rng(seed).permutation(len(self.rows))[:limit]
            self.rows, self.labels = self.rows[pick], self.labels[pick]
        self.train_aug = train_aug
        self.rng = np.random.default_rng(seed)
        self._maps = None  # opened on first access, so DataLoader workers do not copy it

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        if self._maps is None:
            self._maps = np.load(self.maps_path, mmap_mode="r")
        x = encode(self._maps[self.rows[i]], self.channels)
        if self.train_aug:
            x = augment(x, self.rng)
        return torch.from_numpy(x), int(self.labels[i])
