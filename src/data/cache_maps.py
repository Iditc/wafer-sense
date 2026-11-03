"""Cache fixed-size wafer maps as model input.

Maps are resized with nearest-neighbour so the values 0/1/2 stay exact.
The original maps stay in wafers.parquet for yield, Grad-CAM and display.
The cache is written in batches to a memory-mapped .npy, so even all 811K
wafers never have to fit in RAM at once.
Run: python -m src.data.cache_maps [--size 64] [--all]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import PROCESSED_DIR, WAFERS_PARQUET
from src.data.preprocess import resize_nearest

COLUMNS = ["wafer_id", "failure_type", "map_h", "map_w", "map_flat"]
BATCH_SIZE = 20_000


def cache_paths(size: int, subset: str) -> tuple[Path, Path]:
    """Paths of the maps array and the matching wafer_id array."""
    stem = PROCESSED_DIR / f"maps{size}_{subset}"
    return stem.with_suffix(".npy"), Path(f"{stem}_ids.npy")


def count_rows(labeled_only: bool) -> int:
    """Number of wafers that will go into the cache."""
    labels = pd.read_parquet(WAFERS_PARQUET, columns=["failure_type"])["failure_type"]
    return int(labels.notna().sum()) if labeled_only else len(labels)


def build_cache(size: int = 64, labeled_only: bool = True) -> Path:
    """Resize maps to size x size and save them with their wafer ids."""
    subset = "labeled" if labeled_only else "all"
    maps_path, ids_path = cache_paths(size, subset)
    n = count_rows(labeled_only)
    maps = np.lib.format.open_memmap(maps_path, mode="w+", dtype=np.uint8, shape=(n, size, size))
    ids = np.empty(n, dtype=np.int64)

    i = 0
    for batch in pq.ParquetFile(WAFERS_PARQUET).iter_batches(BATCH_SIZE, columns=COLUMNS):
        df = batch.to_pandas()
        if labeled_only:
            df = df[df["failure_type"].notna()]
        for row in df.itertuples(index=False):
            wafer_map = np.asarray(row.map_flat, dtype=np.uint8).reshape(row.map_h, row.map_w)
            maps[i] = resize_nearest(wafer_map, size)
            ids[i] = row.wafer_id
            i += 1
    maps.flush()
    np.save(ids_path, ids)
    return maps_path


def load_cache(size: int = 64, subset: str = "labeled") -> tuple[np.ndarray, np.ndarray]:
    """Load cached maps (memory-mapped, read-only) and their wafer ids."""
    maps_path, ids_path = cache_paths(size, subset)
    return np.load(maps_path, mmap_mode="r"), np.load(ids_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--all", action="store_true", help="include unlabeled wafers")
    args = parser.parse_args()
    path = build_cache(args.size, labeled_only=not args.all)
    maps, ids = load_cache(args.size, "all" if args.all else "labeled")
    print(f"Saved {path.name}: {maps.shape}, values {np.unique(maps[:1000])}")
