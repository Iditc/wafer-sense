"""Pattern-level EDA: representative examples and map sizes."""
import matplotlib.pyplot as plt
import pandas as pd

from src.config import RANDOM_SEED, WAFERS_PARQUET
from src.data.load import map_from_row
from src.data.plot_utils import CLASSES, FONT_SIZE, defect_fraction, display, draw_map, save


def representative_examples(maps: pd.DataFrame, n: int = 3, pool: int = 500) -> pd.DataFrame:
    """Per class, pick n wafers whose defect share is closest to the class median (typical, not extreme)."""
    picks = []
    for cls in CLASSES:
        group = maps[maps["failure_type"] == cls]
        group = group.sample(min(pool, len(group)), random_state=RANDOM_SEED).copy()
        group["frac"] = group.apply(lambda r: defect_fraction(map_from_row(r)), axis=1)
        distance = (group["frac"] - group["frac"].median()).abs()
        picks.append(group.loc[distance.nsmallest(n).index])
    return pd.concat(picks)


def plot_examples(examples: pd.DataFrame, n: int = 3) -> None:
    """Grid of typical wafer maps: one column per class, n rows."""
    fig, axes = plt.subplots(n, len(CLASSES), figsize=(len(CLASSES) * 2.9, n * 3.1))
    for c, cls in enumerate(CLASSES):
        rows = examples[examples["failure_type"] == cls]
        for r in range(n):
            draw_map(axes[r, c], map_from_row(rows.iloc[r]))
        axes[0, c].set_title(display(cls), fontsize=FONT_SIZE + 2)
    fig.suptitle("Typical wafer map of each class (yellow = defective die)", y=1.02)
    save(fig, "02_example_maps.png")


def plot_map_sizes(meta: pd.DataFrame) -> dict:
    """Show small / median / large die grids side by side, plus the most common grid shapes."""
    quantiles = {"Small (5%)": 0.05, "Median": 0.5, "Large (95%)": 0.95}
    ids = [int((meta["die_size"] - meta["die_size"].quantile(q)).abs().idxmin())
           for q in quantiles.values()]
    examples = pd.read_parquet(WAFERS_PARQUET, filters=[("wafer_id", "in", ids)]) \
        .set_index("wafer_id").loc[ids]
    shapes = meta.groupby(["map_h", "map_w"]).size().nlargest(8)

    fig = plt.figure(figsize=(26, 7.5))
    grid = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.7], wspace=0.3)
    for i, (name, (_, row)) in enumerate(zip(quantiles, examples.iterrows())):
        title = f"{name}\n{row['map_h']}x{row['map_w']} grid\n{int(row['die_size']):,} dies"
        draw_map(fig.add_subplot(grid[0, i]), map_from_row(row), title)
    ax = fig.add_subplot(grid[0, 3])
    ax.barh([f"{h}x{w}" for h, w in shapes.index][::-1], shapes.values[::-1], color="steelblue")
    ax.set_title("8 most common grids (of 632)")
    ax.set_xlabel("Wafers")
    fig.suptitle("There are 632 different grid shapes", y=1.05)
    save(fig, "03_map_sizes.png")
    return {"distinct_shapes": int(meta.groupby(["map_h", "map_w"]).ngroups),
            "dies_per_wafer_pct": meta["die_size"].quantile([0.05, 0.5, 0.95]).astype(int).tolist()}


def run(meta: pd.DataFrame, maps: pd.DataFrame) -> dict:
    """Run pattern-level EDA and return summary numbers."""
    plot_examples(representative_examples(maps))
    return plot_map_sizes(meta)
