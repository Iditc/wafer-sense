"""Shared plotting helpers and constants for EDA figures."""
import matplotlib.pyplot as plt
import numpy as np

from src.config import FIGURES_DIR

DEFECTS = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch"]
CLASSES = ["none"] + DEFECTS
DISPLAY_NAMES = {"none": "O.K.", None: "unlabeled"}
FONT_SIZE = 20  # 2x matplotlib's default of 10


def display(cls: str | None) -> str:
    """Human-readable class name for figures ('none' is shown as 'O.K.')."""
    return DISPLAY_NAMES.get(cls, cls)


def setup_style() -> None:
    """Apply large, readable fonts to all figures."""
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE + 2,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 2,
        "ytick.labelsize": FONT_SIZE - 2,
        "legend.fontsize": FONT_SIZE - 2,
        "figure.titlesize": FONT_SIZE + 4,
    })


def save(fig: plt.Figure, name: str) -> None:
    """Save a figure to the figures directory and close it."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / name, dpi=110, bbox_inches="tight")
    plt.close(fig)


def draw_map(ax: plt.Axes, wafer_map: np.ndarray, title: str | None = None) -> None:
    """Draw a single wafer map (0 outside, 1 normal die, 2 defective die)."""
    ax.imshow(wafer_map, cmap="viridis", vmin=0, vmax=2, interpolation="nearest")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=FONT_SIZE)


def defect_fraction(wafer_map: np.ndarray) -> float:
    """Share of dies on the wafer that are defective."""
    dies = (wafer_map > 0).sum()
    return float((wafer_map == 2).sum() / dies) if dies else 0.0
