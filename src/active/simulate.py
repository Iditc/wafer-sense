"""Day 6: simulated active learning on the labeled train wafers.

The train labels start hidden. The first 1% (the same wafers cnn_frac01 was
trained on, so no label leaks into the embedding) is revealed; then each round
a strategy picks `batch` more wafers, the oracle reveals their labels, a linear
probe is refit on the frozen cnn_frac01 embedding and scored on the fixed val set.
Seeds change only the selection, never the start set.
Strategies are in src/active/strategies.py.
Run: python -m src.active.simulate --strategy entropy --rounds 10
"""

import argparse

import numpy as np

from src.active.strategies import STRATEGIES, Round, Strategy
from src.models.dataset import sample_labels
from src.models.embed_eval import fit_probe
from src.models.metrics import evaluate, plot_confusion, save_metrics, summary_line
from src.openset.compare_scores import run_embeddings

LEARNER = "cnn_frac01"  # its embedding saw only the start labels
START_FRAC = 0.01
START_SEED = 0  # the seed cnn_frac01's labels were drawn with
BATCH = 1210  # wafers labeled per round, about 1% of train


class Oracle:
    """Holds the hidden train labels; reveals each wafer's label at most once."""

    def __init__(self, labels: np.ndarray):
        self._labels = labels
        self.revealed = np.zeros(len(labels), dtype=bool)

    def reveal(self, rows: np.ndarray) -> None:
        """Mark rows as labeled; a row asked twice is a bug in the strategy."""
        if len(np.unique(rows)) != len(rows) or self.revealed[rows].any():
            raise ValueError("A wafer was selected twice")
        self.revealed[rows] = True

    def labeled(self) -> tuple[np.ndarray, np.ndarray]:
        """Rows revealed so far and their labels (the only labels anyone sees)."""
        rows = np.flatnonzero(self.revealed)
        return rows, self._labels[rows]


def simulate(
    train: tuple,
    val: tuple,
    strategy: Strategy,
    rounds: int,
    batch: int = BATCH,
    seed: int = 0,
) -> list[dict]:
    """Metrics after the start set and after each round of `batch` new labels."""
    x, y = train
    oracle = Oracle(y)
    oracle.reveal(sample_labels(y, START_FRAC, START_SEED))
    rng = np.random.default_rng(seed)
    history = []
    for r in range(rounds + 1):
        rows, labels = oracle.labeled()
        model = fit_probe(x[rows], labels)
        history.append(
            {
                "round": r,
                "n_labels": len(rows),
                **evaluate(val[1], model.predict(val[0])),
            }
        )
        print(
            f"round {r} ({len(rows):,} labels): {summary_line(history[-1])}", flush=True
        )
        unlabeled = np.flatnonzero(~oracle.revealed)
        if r == rounds or len(unlabeled) == 0:
            break
        n = min(batch, len(unlabeled))
        oracle.reveal(strategy(Round(x, model, rows, unlabeled, n, rng)))
    return history


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--strategy", default="random", choices=sorted(STRATEGIES))
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    """Run one strategy, save the curve and the first and last confusion matrices."""
    args = parse_args()
    emb = run_embeddings(LEARNER)
    history = simulate(
        emb["train"],
        emb["val"],
        STRATEGIES[args.strategy],
        args.rounds,
        args.batch,
        args.seed,
    )
    tag = f"al_{args.strategy}_seed{args.seed}"
    save_metrics({"learner": LEARNER, "config": vars(args), "history": history}, tag)
    for h in (history[0], history[-1]):
        plot_confusion(
            h["confusion_matrix"],
            f"{args.strategy}, {h['n_labels']:,} labels: F1 macro {h['f1_macro']:.3f}",
            f"{tag}_round{h['round']}_confusion.png",
        )


if __name__ == "__main__":
    main()
