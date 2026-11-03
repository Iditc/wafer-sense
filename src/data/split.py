"""Lot-based train/val/test split, fixed for the whole project.

Every lot goes to exactly one split, so near-identical wafers of one lot never
end up on both sides. Lots are assigned with StratifiedGroupKFold so each split
keeps a similar mix of classes and lot types (full / mixed / unlabeled).
Run: python -m src.data.split
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import METRICS_DIR, RANDOM_SEED, SPLIT_PARQUET, WAFERS_PARQUET
from src.data.plot_utils import CLASSES, display, save, setup_style

N_FOLDS = 20
SPLIT_FOLDS = {"train": range(0, 14), "val": range(14, 17), "test": range(17, 20)}  # 70/15/15
SPLITS = list(SPLIT_FOLDS)
META_COLS = ["wafer_id", "lot_name", "failure_type", "die_size"]


def add_lot_type(meta: pd.DataFrame) -> pd.DataFrame:
    """Add a lot_type column: 'full', 'mixed' or 'unlabeled', based on the lot's labeled share."""
    frac = meta["failure_type"].notna().groupby(meta["lot_name"]).transform("mean")
    meta["lot_type"] = np.select([frac == 1, frac == 0], ["full", "unlabeled"], "mixed")
    return meta


def assign_splits(meta: pd.DataFrame) -> pd.Series:
    """Assign every wafer a split via its lot, stratified by lot type and class."""
    strata = meta["lot_type"] + "|" + meta["failure_type"].fillna("unlabeled")
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold = np.empty(len(meta), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(meta, strata, groups=meta["lot_name"])):
        fold[idx] = k
    fold_to_split = {k: name for name, folds in SPLIT_FOLDS.items() for k in folds}
    return pd.Series(fold, index=meta.index).map(fold_to_split)


def check_no_leakage(meta: pd.DataFrame) -> None:
    """Fail loudly if any lot appears in more than one split."""
    splits_per_lot = meta.groupby("lot_name")["split"].nunique()
    leaked = int((splits_per_lot > 1).sum())
    if leaked:
        raise AssertionError(f"{leaked} lots appear in more than one split")


def class_share(meta: pd.DataFrame) -> pd.DataFrame:
    """Per class (incl. unlabeled): share of its wafers in each split, rows sum to 100%."""
    cls = meta["failure_type"].fillna("unlabeled")
    table = pd.crosstab(cls, meta["split"], normalize="index")[SPLITS] * 100
    return table.loc[CLASSES + ["unlabeled"]]


def plot_class_share(share: pd.DataFrame) -> None:
    """Stacked bars: how each class is divided between train / val / test."""
    colors = {"train": "steelblue", "val": "orange", "test": "seagreen"}
    labels = [display(c) if c != "unlabeled" else "Unlabeled" for c in share.index][::-1]
    fig, ax = plt.subplots(figsize=(15, 9))
    left = np.zeros(len(share))
    for split in SPLITS:
        values = share[split].values[::-1]
        ax.barh(labels, values, left=left, color=colors[split], label=split)
        for y, (x0, v) in enumerate(zip(left, values)):
            ax.text(x0 + v / 2, y, f"{v:.0f}%", ha="center", va="center", color="white")
        left += values
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of the class's wafers (%)")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    ax.set_title("Each class is split ~70 / 15 / 15 across train / val / test", pad=60)
    save(fig, "10_split_class_share.png")


def summarize(meta: pd.DataFrame) -> dict:
    """Wafer, lot, class and lot-type counts per split, plus median die count."""
    by_split = meta.groupby("split")
    return {
        "wafers": by_split.size()[SPLITS].astype(int).to_dict(),
        "lots": by_split["lot_name"].nunique()[SPLITS].astype(int).to_dict(),
        "classes": pd.crosstab(meta["failure_type"], meta["split"])[SPLITS].astype(int).to_dict(),
        "lot_types": pd.crosstab(meta["lot_type"], meta["split"])[SPLITS].astype(int).to_dict(),
        "median_die_size": by_split["die_size"].median()[SPLITS].to_dict(),
    }


def load_split() -> pd.DataFrame:
    """Load the saved split: one row per wafer with wafer_id, lot_name, lot_type, split."""
    return pd.read_parquet(SPLIT_PARQUET)


def main() -> None:
    """Build the split, verify it, save it, and plot the class balance."""
    setup_style()
    meta = add_lot_type(pd.read_parquet(WAFERS_PARQUET, columns=META_COLS))
    meta["split"] = assign_splits(meta)
    check_no_leakage(meta)
    meta[["wafer_id", "lot_name", "lot_type", "split"]].to_parquet(SPLIT_PARQUET, index=False)

    share = class_share(meta)
    plot_class_share(share)
    summary = summarize(meta)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(share.round(1))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
