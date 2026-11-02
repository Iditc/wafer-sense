"""Map-level preprocessing utilities shared by EDA and model training."""
import numpy as np


def resize_nearest(wafer_map: np.ndarray, size: int) -> np.ndarray:
    """Resize a wafer map to size x size with nearest-neighbour sampling.

    Nearest-neighbour keeps the discrete values (0/1/2) intact, unlike interpolation.
    """
    h, w = wafer_map.shape
    rows = (np.arange(size) * h / size).astype(int)
    cols = (np.arange(size) * w / size).astype(int)
    return wafer_map[np.ix_(rows, cols)]
