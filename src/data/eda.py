"""Exploratory data analysis for WM-811K: entry point and class-level plots.

Pattern-level plots live in eda_patterns.py, lot-level plots in eda_lots.py.
Run: python -m src.data.eda
"""
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from src.config import METRICS_DIR, WAFERS_PARQUET
from src.data import eda_lots, eda_patterns
from src.data.load import load_wafers
from src.data.plot_utils import CLASSES, DEFECTS, FONT_SIZE, display, save, setup_style

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
                f" {n / 1000:.0f}K ({n / len(meta):.0%})", va="center")
    ax.set_xlim(0, n_unl * 1.35)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:.0f}K"))
    ax.set_xlabel("Wafers")
    ax.set_title("Only 21% of wafers are labeled")
    save(fig, "01a_labeled_share.png")
    return {"labeled": n_lab, "unlabeled": n_unl}


def plot_class_distribution(meta: pd.DataFrame) -> dict:
    """Bar chart of labeled classes on a log scale."""
    counts = meta["failure_type"].value_counts()
    counts.index = [display(c) for c in counts.index]
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.barplot(x=counts.index, y=counts.values, ax=ax, color="steelblue")
    ax.set_yscale("log")
    ax.set_title("Labeled classes: ~1000x gap between O.K. and Near-full")
    ax.set_ylabel("Log number of wafers")
    ax.set_xlabel("Pattern")
    ax.tick_params(axis="x", rotation=35)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=FONT_SIZE - 4)
    save(fig, "01_class_distribution.png")
    return {k: int(v) for k, v in counts.items()}


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
    meta = load_meta()
    summary = {"labeled_share": plot_labeled_share(meta),
               "class_counts": plot_class_distribution(meta),
               "lots": eda_lots.run(meta)}
    plot_defect_by_index(meta)

    maps = load_maps(CLASSES)
    summary["map_sizes"] = eda_patterns.run(meta, maps)
    eda_lots.plot_label_noise(maps)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
