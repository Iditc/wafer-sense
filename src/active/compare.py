"""Day 8: full active-learning simulation and the F1-vs-labels curves.

Random selection runs with 3 seeds. Entropy, margin and k-center have no
randomness (the start set is fixed, so every seed would pick the same wafers),
so they run once; random's seed spread is the noise yardstick for them.
Reported per strategy: labels needed to reach F1 TARGET, F1 and missed-defect
rate at each budget. Finished runs are reused (k-center takes ~25 min).
Run: python -m src.active.compare
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.active.simulate import LEARNER, run
from src.active.strategies import STRATEGIES
from src.config import METRICS_DIR
from src.data.plot_utils import save, setup_style
from src.openset.compare_scores import run_embeddings

ROUNDS = 10
TARGET = 0.75  # F1 macro; the frozen probe's ceiling is about 0.80
RANDOM_SEEDS = (0, 1, 2)
COLORS = {
    "random": "#7f7f7f",
    "entropy": "#d9a441",
    "margin": "#1f4d3a",
    "kcenter": "#3fa9b5",
}


def runs_to_do() -> list[tuple[str, int]]:
    """(strategy, seed) pairs: random with every seed, the others once."""
    return [
        (s, seed)
        for s in STRATEGIES
        for seed in (RANDOM_SEEDS if s == "random" else (0,))
    ]


def load_or_run(strategy: str, seed: int, emb: dict) -> list[dict]:
    """A saved curve with ROUNDS rounds, or a fresh simulation."""
    path = METRICS_DIR / f"al_{strategy}_seed{seed}.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if len(saved["history"]) == ROUNDS + 1:
            return saved["history"]
    print(f"--- {strategy} seed {seed}", flush=True)
    return run(strategy, ROUNDS, seed=seed, emb=emb)


def labels_to_reach(history: list[dict], target: float = TARGET) -> int | None:
    """Labels used in the first round with F1 macro >= target (None if never)."""
    hits = [h["n_labels"] for h in history if h["f1_macro"] >= target]
    return hits[0] if hits else None


def curve(histories: list[list[dict]], key: str) -> tuple[np.ndarray, np.ndarray]:
    """Label counts and one row per run of a metric (runs share the same budgets)."""
    labels = np.array([h["n_labels"] for h in histories[0]])
    return labels, np.array([[h[key] for h in run_] for run_ in histories])


def summarize(results: dict[str, list[list[dict]]]) -> dict:
    """JSON-ready table: labels to reach TARGET and the metrics at each budget."""
    table = {}
    for s, histories in results.items():
        labels, f1 = curve(histories, "f1_macro")
        _, missed = curve(histories, "missed_defect_rate")
        reach = [labels_to_reach(h) for h in histories]
        table[s] = {
            "runs": len(histories),
            "labels_to_reach_target": reach,
            "n_labels": labels.tolist(),
            "f1_mean": np.round(f1.mean(axis=0), 4).tolist(),
            "f1_min": np.round(f1.min(axis=0), 4).tolist(),
            "f1_max": np.round(f1.max(axis=0), 4).tolist(),
            "missed_mean": np.round(missed.mean(axis=0), 4).tolist(),
        }
    return table


def plot_curves(table: dict) -> None:
    """F1 macro and missed-defect rate vs labels used, one line per strategy."""
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(22, 8))
    for s, row in table.items():
        x = np.array(row["n_labels"]) / 1000
        label = f"{s} ({row['runs']} seeds)" if row["runs"] > 1 else s
        axes[0].plot(x, row["f1_mean"], "o-", color=COLORS[s], lw=3, label=label)
        if row["runs"] > 1:
            axes[0].fill_between(
                x, row["f1_min"], row["f1_max"], color=COLORS[s], alpha=0.25
            )
        axes[1].plot(x, np.array(row["missed_mean"]) * 100, "o-", color=COLORS[s], lw=3)
    axes[0].axhline(TARGET, color="#c2185b", ls="--", lw=2, label=f"target {TARGET}")
    axes[0].set(xlabel="Labels used (thousands)", ylabel="F1 macro (val)")
    axes[0].set_title("F1 macro vs labels")
    axes[0].legend(loc="lower right")
    axes[1].set(xlabel="Labels used (thousands)", ylabel="Missed defects (%)")
    axes[1].set_title("Defects called O.K. vs labels")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(f"Simulated active learning, probe on frozen {LEARNER} embedding")
    fig.tight_layout()
    save(fig, "al_curves.png")


def main() -> None:
    """Run every strategy, save the comparison table and the curves figure."""
    torch.set_num_threads(os.cpu_count() or 1)
    emb = run_embeddings(LEARNER)
    results: dict[str, list] = {s: [] for s in STRATEGIES}
    for s, seed in runs_to_do():
        results[s].append(load_or_run(s, seed, emb))
    table = summarize(results)
    for s, row in table.items():
        print(s, row["labels_to_reach_target"], row["f1_mean"][-1], flush=True)
    out = {"learner": LEARNER, "rounds": ROUNDS, "target": TARGET, "strategies": table}
    (METRICS_DIR / "al_compare_val.json").write_text(json.dumps(out, indent=2) + "\n")
    plot_curves(table)


if __name__ == "__main__":
    main()
