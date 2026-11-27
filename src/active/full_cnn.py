"""Day 8b: does the active-learning winner also help a full CNN?

Days 6-8 used a fast learner (a probe on the frozen cnn_frac01 embedding).
Here the label sets that margin and random picked in that simulation are used
to train the full CNN from scratch on Colab, at four label budgets.
The selection still comes from the fast learner; only the training changes.
That checks the label sets, not a full-CNN selection loop (too costly).

Locally:  python -m src.active.full_cnn --export   (writes the label sets)
Colab:    python -m src.active.full_cnn             (prints the runs)
Locally:  python -m src.active.full_cnn --summary  (table and figure)
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from src.config import METRICS_DIR
from src.data.plot_utils import save, setup_style

STRATEGIES = ("margin", "random")
SEED = 0
BUDGETS = (2420, 4840, 8470, 13310)
# Epochs and patience per budget: the few-label recipes of week 2
# (1% labels: 200 / 40, 10%: 60 / 15), scaled in between.
EPOCHS = {2420: (200, 40), 4840: (120, 30), 8470: (80, 20), 13310: (60, 15)}


def selection_path(strategy: str):
    """Where a strategy's picks (start set + one list per round) are stored."""
    return METRICS_DIR / f"al_rows_{strategy}_seed{SEED}.json"


def run_name(strategy: str, budget: int) -> str:
    """Checkpoint and metrics name of one full-CNN run."""
    return f"cnn_al_{strategy}_{budget}"


def export_selection(strategy: str, rounds: int) -> dict:
    """Rerun the fast simulation and save which train rows were picked, in order."""
    from src.active.simulate import LEARNER, START_FRAC, START_SEED, simulate
    from src.active.strategies import STRATEGIES as ALL
    from src.models.dataset import sample_labels
    from src.openset.compare_scores import run_embeddings

    emb = run_embeddings(LEARNER)
    picks = []

    def record(state):
        pick = ALL[strategy](state)
        picks.append(pick)
        return pick

    simulate(emb["train"], emb["val"], record, rounds, seed=SEED)
    start = sample_labels(emb["train"][1], START_FRAC, START_SEED)
    out = {"start": start.tolist(), "rounds": [p.tolist() for p in picks]}
    selection_path(strategy).write_text(json.dumps(out) + "\n")
    return out


def budget_rows(
    strategy: str, budget: int, selection: dict | None = None
) -> np.ndarray:
    """Train rows of the first `budget` labels: the start set, then whole rounds."""
    sel = selection or json.loads(selection_path(strategy).read_text())
    rows = list(sel["start"])
    for pick in sel["rounds"]:
        if len(rows) >= budget:
            break
        rows += pick
    if len(rows) != budget:
        raise ValueError(f"{budget} is not a round boundary of {strategy}")
    return np.array(rows)


def colab_runs() -> list[str]:
    """One line per run: name, strategy, budget, epochs, patience."""
    return [
        f"{run_name(s, b)} {s} {b} {EPOCHS[b][0]} {EPOCHS[b][1]}"
        for b in BUDGETS
        for s in STRATEGIES
    ]


def load_curves() -> dict[str, list[tuple[int, float, float]]]:
    """(budget, F1 macro, missed-defect rate) of every finished full-CNN run."""
    curves = {s: [] for s in STRATEGIES}
    for s in STRATEGIES:
        for b in BUDGETS:
            path = METRICS_DIR / f"{run_name(s, b)}.json"
            if path.exists():
                m = json.loads(path.read_text())
                curves[s].append((b, m["f1_macro"], m["missed_defect_rate"]))
    return curves


def plot_summary(curves: dict) -> None:
    """Full-CNN F1 vs labels, with the fast learner's curves dashed for reference."""
    setup_style()
    probe = json.loads((METRICS_DIR / "al_compare_val.json").read_text())["strategies"]
    colors = {"margin": "#1f4d3a", "random": "#7f7f7f"}
    fig, ax = plt.subplots(figsize=(13, 8))
    for s in STRATEGIES:
        if curves[s]:
            b, f1, _ = zip(*curves[s])
            ax.plot(
                np.array(b) / 1000,
                f1,
                "o-",
                lw=3,
                color=colors[s],
                label=f"{s}, full CNN",
            )
        x = np.array(probe[s]["n_labels"]) / 1000
        ax.plot(
            x, probe[s]["f1_mean"], "--", lw=2, color=colors[s], label=f"{s}, probe"
        )
    ax.set(xlabel="Labels used (thousands)", ylabel="F1 macro (val)")
    ax.set_title("Same label sets: full CNN vs frozen probe")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    save(fig, "al_full_cnn.png")


def main() -> None:
    """--export writes the label sets, --summary compares, default prints the runs."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--export", action="store_true")
    p.add_argument("--summary", action="store_true")
    args = p.parse_args()
    if args.export:
        rounds = (max(BUDGETS) - 1210) // 1210
        for s in STRATEGIES:
            export_selection(s, rounds)
    elif args.summary:
        curves = load_curves()
        for s, rows in curves.items():
            print(s, rows)
        out = {"strategies": STRATEGIES, "budgets": BUDGETS, "curves": curves}
        (METRICS_DIR / "al_full_cnn_val.json").write_text(
            json.dumps(out, indent=2) + "\n"
        )
        plot_summary(curves)
    else:
        print("\n".join(colab_runs()))


if __name__ == "__main__":
    main()
