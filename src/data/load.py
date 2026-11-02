"""Load WM-811K from the raw pickle, clean it, and cache it as parquet.

Wafer map values: 0 = outside wafer, 1 = normal die, 2 = defective die.
Maps have different shapes, so each map is stored flattened with its height/width.
"""
import sys

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RAW_PICKLE, WAFERS_PARQUET


def _register_old_pandas_aliases() -> None:
    """Map module paths from old pandas (used when LSWMD.pkl was saved) to new ones."""
    import pandas.core.indexes as new_indexes
    import pandas.core.indexes.base as new_base

    sys.modules.setdefault("pandas.indexes", new_indexes)
    sys.modules.setdefault("pandas.indexes.base", new_base)


def read_raw_pickle() -> pd.DataFrame:
    """Read the raw LSWMD.pkl, retrying with compatibility aliases if needed."""
    if not RAW_PICKLE.exists():
        raise FileNotFoundError(f"Raw data not found: {RAW_PICKLE}")
    try:
        return pd.read_pickle(RAW_PICKLE)
    except ModuleNotFoundError:
        _register_old_pandas_aliases()
        return pd.read_pickle(RAW_PICKLE)


def _unwrap_label(value) -> str | None:
    """Extract a string label from the nested array format, e.g. [['Scratch']] -> 'Scratch'."""
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    return str(arr.ravel()[0])


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw columns into flat, typed columns ready for parquet."""
    maps = df["waferMap"].map(lambda m: np.asarray(m, dtype=np.uint8))
    return pd.DataFrame({
        "wafer_id": np.arange(len(df)),
        "lot_name": df["lotName"].astype(str).values,
        "wafer_index": df["waferIndex"].astype(int).values,
        "die_size": df["dieSize"].astype(float).values,
        "failure_type": df["failureType"].map(_unwrap_label).values,
        "orig_split": df["trianTestLabel"].map(_unwrap_label).values,
        "map_h": maps.map(lambda m: m.shape[0]).values,
        "map_w": maps.map(lambda m: m.shape[1]).values,
        "map_flat": maps.map(lambda m: m.ravel()).values,
    })


def map_from_row(row) -> np.ndarray:
    """Rebuild the 2D wafer map from a parquet row."""
    return np.asarray(row["map_flat"], dtype=np.uint8).reshape(row["map_h"], row["map_w"])


def load_wafers() -> pd.DataFrame:
    """Load processed wafers; build the parquet from the raw pickle only if missing."""
    if WAFERS_PARQUET.exists():
        return pd.read_parquet(WAFERS_PARQUET)
    df = clean(read_raw_pickle())
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(WAFERS_PARQUET, index=False)
    return df


if __name__ == "__main__":
    wafers = load_wafers()
    print(f"Wafers: {len(wafers):,} | Lots: {wafers['lot_name'].nunique():,}")
    print(wafers["failure_type"].value_counts(dropna=False))
