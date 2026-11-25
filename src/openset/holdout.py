"""Leave-one-type-out runs: one CNN per defect type, trained without that type.

Each run uses the cnn_cosine recipe with --exclude <type>, so the type is absent
from train and val. Day 5 scores how well each model flags its missing type as new.
Print the runs for Colab: python -m src.openset.holdout
"""

from src.data.plot_utils import DEFECTS

# Held-out runs that already exist (cnn_noscratch, week 2).
TRAINED = ("Scratch",)


def run_name(defect: str) -> str:
    """Checkpoint and metrics name of the run that never sees one defect type."""
    if defect not in DEFECTS:
        raise ValueError(f"Unknown defect type '{defect}'")
    return "cnn_no" + defect.lower().replace("-", "")


def holdout_runs(skip: tuple[str, ...] = TRAINED) -> dict[str, str]:
    """Run name -> held-out type, for every defect type not in skip."""
    return {run_name(d): d for d in DEFECTS if d not in skip}


if __name__ == "__main__":
    for name, defect in holdout_runs().items():
        print(name, defect)
