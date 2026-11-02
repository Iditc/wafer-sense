"""Exploratory data analysis for WM-811K: classes, example maps, map sizes, wafer index.

Lot-level analysis and label noise live in eda_lots.py. Run: python -m src.data.eda
"""
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import METRICS_DIR, RANDOM_SEED, WAFERS_PARQUET
from src.data import eda_lots
from src.data.load import load_wafers, map_from_row
from src.data.plot_utils import DEFECTS, FONT_SIZE, draw_map, save, setup_style

META_COLS = ["wafer_id", "lot_name", "wafer_index", "die_size", "failure_type", "map_h", "map_w"]


def load_meta() -> pd.DataFrame:
    """Load all columns except the heavy wafer maps."""
    load_wafers()  # make sure the parquet cache exists
    return pd.read_parquet(WAFERS_PARQUET, columns=META_COLS)


def load_maps(classes: list[str]) -> pd.DataFrame:
    """Load full rows (incl. maps) for the given failure types only."""
    return pd.read_parquet(WAFERS_PARQUET, filters=[("failure_type", "in", classes)])


def plot_labeled_share(meta: pd.DataFrame) -> dict:
    """Bar chart: labeled vs unlabeled wafers."""
    n_lab = int(meta["failure_type"].notna().sum())
    n_unl = int(len(meta) - n_lab)
    fig, ax = plt.subplots(figsize=(11, 4))
    bars = ax.barh(["Unlabeled", "Labeled"], [n_unl, n_lab], color=["lightgray", "steelblue"])
    for bar, n in zip(bars, [n_unl, n_lab]):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f" {n:,} ({n / len(meta):.0%})", va="center")
    ax.set_xlim(0, n_unl * 1.35)
    ax.set_xlabel("Wafers")
    ax.set_title("Only 21% of wafers are labeled")
    save(fig, "01a_labeled_share.png")
    return {"labeled": n_lab, "unlabeled": n_unl}


def plot_class_distribution(meta: pd.DataFrame) -> dict:
    """Bar chart of labeled classes on a log scale."""
    counts = meta["failure_type"].value_counts()
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.barplot(x=counts.index, y=counts.values, ax=ax, color="steelblue")
    ax.set_yscale("log")
    ax.set_title("Labeled classes (log scale): 65x gap between Edge-Ring and Near-full")
    ax.set_ylabel("Wafers")
    ax.set_xlabel("Pattern")
    ax.tick_params(axis="x", rotation=35)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=FONT_SIZE - 4)
    save(fig, "01_class_distribution.png")
    return {k: int(v) for k, v in counts.items()}


def plot_examples(samples: pd.DataFrame, n_per_class: int) -> None:
    """Grid of example wafer maps: one row per class."""
    classes = DEFECTS + ["none"]
    fig, axes = plt.subplots(len(classes), n_per_class,
                             figsize=(n_per_class * 2.6, len(classes) * 2.6))
    for r, cls in enumerate(classes):
        rows = samples[samples["failure_type"] == cls]
        for c in range(n_per_class):
            axes[r, c].axis("off")
            if c < len(rows):
                draw_map(axes[r, c], map_from_row(rows.iloc[c]))
        axes[r, 0].text(-0.1, 0.5, cls, transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=FONT_SIZE + 2)
    fig.suptitle("Example wafer maps (yellow = defective die)", y=1.0)
    save(fig, "02_example_maps.png")


def plot_map_sizes(meta: pd.DataFrame) -> dict:
    """Histogram of map height/width and the most common shapes."""
    shapes = meta.groupby(["map_h", "map_w"]).size().nlargest(8)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    ax1.hist(meta["map_h"], bins=60, alpha=0.6, label="height")
    ax1.hist(meta["map_w"], bins=60, alpha=0.6, label="width")
    ax1.set_yscale("log")
    ax1.set_title("Map dimensions (dies)")
    ax1.set_xlabel("Dies")
    ax1.legend()
    labels = [f"{h}x{w}" for h, w in shapes.index]
    ax2.barh(labels[::-1], shapes.values[::-1], color="steelblue")
    ax2.set_title("8 most common shapes (of 632)")
    ax2.set_xlabel("Wafers")
    save(fig, "03_map_sizes.png")
    return {"distinct_shapes": int(meta.groupby(["map_h", "map_w"]).ngroups),
            "height_pct": meta["map_h"].quantile([0.05, 0.5, 0.95]).astype(int).tolist(),
            "width_pct": meta["map_w"].quantile([0.05, 0.5, 0.95]).astype(int).tolist()}


def plot_defect_by_index(meta: pd.DataFrame) -> None:
    """Defect rate by wafer position within the lot, on a 0-based axis to avoid exaggeration."""
    lab = meta[meta["failure_type"].notna()]
    rate = lab.groupby("wafer_index")["failure_type"].apply(lambda s: s.isin(DEFECTS).mean())
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(rate.index, rate.values * 100, marker="o", linewidth=2.5, markersize=9)
    ax.set_ylim(0, 25)
    ax.set_title("Defect rate is flat across wafer position in the lot")
    ax.set_xlabel("Wafer index within lot")
    ax.set_ylabel("Defective wafers (%)")
    save(fig, "04_defect_rate_by_index.png")


def main() -> None:
    """Run all EDA steps and save figures + a JSON summary."""
    setup_style()
    n_examples = 6
    meta = load_meta()
    summary = {"labeled_share": plot_labeled_share(meta),
               "class_counts": plot_class_distribution(meta),
               "map_sizes": plot_map_sizes(meta),
               "lots": eda_lots.run(meta)}
    plot_defect_by_index(meta)

    maps = load_maps(DEFECTS + ["none"])
    samples = maps.groupby("failure_type", group_keys=False)[maps.columns] \
        .apply(lambda g: g.sample(min(n_examples, len(g)), random_state=RANDOM_SEED))
    plot_examples(samples, n_examples)
    eda_lots.plot_label_noise(maps)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
