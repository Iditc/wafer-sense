"""Figures for the day 5 new-type score comparison."""

import matplotlib.pyplot as plt
import numpy as np

from src.data.plot_utils import save, setup_style

LABELS = {
    "softmax": "Max softmax",
    "energy": "Energy",
    "knn": "kNN distance",
    "mahalanobis": "Mahalanobis",
}


def plot_score_auroc(table: dict) -> None:
    """Scores x held-out types: AUROC of new vs known defects, with a macro column."""
    setup_style()
    scores = list(table)
    types = list(table[scores[0]]["per_type"])
    columns = types + ["Macro"]
    values = np.array(
        [
            [table[s]["per_type"][t]["auroc"] for t in types]
            + [table[s]["macro_auroc"]]
            for s in scores
        ]
    )
    fig, ax = plt.subplots(figsize=(20, 7))
    ax.imshow(values, cmap="Blues", vmin=0.5, vmax=1, aspect="auto")
    for i in range(len(scores)):
        for j in range(len(columns)):
            color = "white" if values[i, j] > 0.8 else "black"
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", color=color)
    ax.axvline(len(types) - 0.5, color="black", linewidth=3)
    ax.set_xticks(range(len(columns)), columns, rotation=35, ha="right")
    ax.set_yticks(range(len(scores)), [LABELS[s] for s in scores])
    ax.set_title("New-type AUROC by score (val, each type held out in its own CNN)")
    save(fig, "openset_scores_auroc.png")
