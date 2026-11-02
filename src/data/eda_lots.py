"""Lot-level EDA (lot size, labeling per lot, pattern repetition) and label-noise examples."""
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import RANDOM_SEED, WAFERS_PARQUET
from src.data.load import map_from_row
from src.data.plot_utils import DEFECTS, FONT_SIZE, defect_fraction, draw_map, save


def dominant_pattern_share(meta: pd.DataFrame) -> pd.Series:
    """For each lot with 2+ defective wafers: share of its most common defect pattern."""
    defects = meta[meta["failure_type"].isin(DEFECTS)]
    sizes = defects.groupby("lot_name").size()
    multi = defects[defects["lot_name"].isin(sizes[sizes >= 2].index)]
    return multi.groupby("lot_name")["failure_type"] \
        .apply(lambda s: s.value_counts(normalize=True).iloc[0])


def labeling_per_lot(meta: pd.DataFrame) -> pd.Series:
    """Count lots that are fully labeled, fully unlabeled, or mixed."""
    frac = meta.groupby("lot_name")["failure_type"].apply(lambda s: s.notna().mean())
    return pd.Series({"Fully unlabeled": int((frac == 0).sum()),
                      "Mixed": int(((frac > 0) & (frac < 1)).sum()),
                      "Fully labeled": int((frac == 1).sum())})


def plot_wafers_per_lot(meta: pd.DataFrame) -> None:
    """Histogram of the number of wafers per lot."""
    sizes = meta.groupby("lot_name").size()
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.hist(sizes, bins=np.arange(0.5, 26.5, 1), color="steelblue")
    ax.set_yscale("log")
    ax.set_title("Most lots hold 25 wafers (standard lot size)")
    ax.set_xlabel("Wafers in lot")
    ax.set_ylabel("Lots (log scale)")
    save(fig, "05_wafers_per_lot.png")


def plot_labeling_per_lot(counts: pd.Series) -> None:
    """Bar chart: labeling status per lot."""
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(counts.index, counts.values, color=["lightgray", "orange", "steelblue"])
    for bar, n in zip(bars, counts.values):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {n:,}", va="center")
    ax.set_xlim(0, counts.max() * 1.25)
    ax.set_title("Labeling was done per lot")
    ax.set_xlabel("Lots")
    save(fig, "06_labeling_per_lot.png")


def plot_dominant_share(dom: pd.Series) -> None:
    """Histogram of the dominant pattern share in lots with 2+ defective wafers."""
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.hist(dom, bins=np.linspace(0.1, 1.0, 19), color="steelblue")
    ax.set_title(f"In {(dom == 1).mean():.0%} of lots, all defects share one pattern")
    ax.set_xlabel("Share of the most common pattern in the lot")
    ax.set_ylabel("Lots")
    save(fig, "07_pattern_repetition.png")


def pick_example_lot(meta: pd.DataFrame, dom: pd.Series) -> str:
    """Pick a fully labeled lot with a single repeated pattern and the most defects."""
    single = dom[dom == 1].index
    lots = meta[meta["lot_name"].isin(single)].groupby("lot_name")
    full = lots["failure_type"].apply(lambda s: s.notna().all())
    n_def = lots["failure_type"].apply(lambda s: s.isin(DEFECTS).sum())
    return n_def[full[full].index].idxmax()


def plot_example_lot(lot_name: str) -> None:
    """Draw every wafer of one lot, titled with its label."""
    lot = pd.read_parquet(WAFERS_PARQUET, filters=[("lot_name", "==", lot_name)]) \
        .sort_values("wafer_index")
    cols = 5
    rows = math.ceil(len(lot) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.8, rows * 4.0))
    for ax, (_, row) in zip(axes.ravel(), lot.iterrows()):
        label = row["failure_type"] or "unlabeled"
        draw_map(ax, map_from_row(row), f"#{row['wafer_index']} {label}")
    for ax in axes.ravel()[len(lot):]:
        ax.axis("off")
    fig.suptitle(f"One lot ({lot_name}): the same pattern repeats across wafers", y=1.0)
    save(fig, "08_example_lot.png")


def plot_label_noise(maps: pd.DataFrame, n: int = 5) -> None:
    """Show 'Center' wafers with the fewest defects and 'none' wafers with the most."""
    center = maps[maps["failure_type"] == "Center"].copy()
    clean = maps[maps["failure_type"] == "none"].sample(20000, random_state=RANDOM_SEED).copy()
    for df in (center, clean):
        df["frac"] = df.apply(lambda r: defect_fraction(map_from_row(r)), axis=1)
    groups = [("Labeled 'Center'\n(weakest: tiny cluster)", center.nsmallest(n, "frac")),
              ("Labeled 'none'\n(looks like Random)", clean.nlargest(n, "frac"))]
    fig, axes = plt.subplots(2, n, figsize=(n * 3.6, 2 * 4.2))
    for r, (name, rows) in enumerate(groups):
        for c, (_, row) in enumerate(rows.iterrows()):
            draw_map(axes[r, c], map_from_row(row), f"{row['frac']:.0%} defective")
        axes[r, 0].text(-0.1, 0.5, name, transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=FONT_SIZE)
    fig.suptitle("Ambiguous labels: very weak patterns vs. heavily defective 'none'", y=1.02)
    save(fig, "09_label_noise.png")


def run(meta: pd.DataFrame) -> dict:
    """Run lot-level analysis, save its figures, and return summary numbers."""
    dom = dominant_pattern_share(meta)
    labeling = labeling_per_lot(meta)
    plot_wafers_per_lot(meta)
    plot_labeling_per_lot(labeling)
    plot_dominant_share(dom)
    plot_example_lot(pick_example_lot(meta, dom))
    sizes = meta.groupby("lot_name").size()
    return {"lots": int(meta["lot_name"].nunique()),
            "wafers_per_lot_pct": sizes.quantile([0.05, 0.5, 0.95]).astype(int).tolist(),
            "labeling": {k: int(v) for k, v in labeling.items()},
            "lots_with_2plus_defects": int(len(dom)),
            "lots_single_pattern_pct": round(float((dom == 1).mean() * 100), 1)}
