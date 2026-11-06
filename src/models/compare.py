"""Compare saved runs and pick the best one under the model selection rule.

Rule: highest val F1 macro among runs whose missed-defect rate is at most a limit.
The limit is a parameter (MAX_MISSED_DEFECT_RATE in config), so this script also
shows which run would win under other limits.
Run: python -m src.models.compare [--limits 0.03 0.04 0.05 0.07 1]
"""
import argparse
import json

import pandas as pd

from src.config import MAX_MISSED_DEFECT_RATE, METRICS_DIR


def load_runs(pattern: str = "cnn_*.json") -> pd.DataFrame:
    """One row per saved run: name, F1 macro, missed-defect and false-alarm rates."""
    rows = []
    for path in sorted(METRICS_DIR.glob(pattern)):
        m = json.loads(path.read_text(encoding="utf-8"))
        if m.get("split") != "val":
            continue
        rows.append({"run": path.stem, "f1_macro": m["f1_macro"],
                     "missed": m["missed_defect_rate"], "false_alarm": m["false_alarm_rate"]})
    return pd.DataFrame(rows).sort_values("f1_macro", ascending=False, ignore_index=True)


def pick_best(runs: pd.DataFrame, max_missed: float = MAX_MISSED_DEFECT_RATE) -> str | None:
    """Name of the run with the highest F1 macro that passes the missed-defect limit."""
    allowed = runs[runs["missed"] <= max_missed]
    return None if allowed.empty else allowed.iloc[0]["run"]


def main() -> None:
    """Print all runs, then the winner under each missed-defect limit."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pattern", default="cnn_*.json", help="which metrics files to compare")
    p.add_argument("--limits", type=float, nargs="+", default=[0.03, 0.04, 0.05, 0.07, 1.0])
    args = p.parse_args()

    runs = load_runs(args.pattern)
    print(runs.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nWinner per missed-defect limit (project default {MAX_MISSED_DEFECT_RATE:.0%}):")
    for limit in args.limits:
        label = "no limit" if limit >= 1 else f"<= {limit:.0%}"
        print(f"  {label:>8}: {pick_best(runs, limit) or 'none passes'}")


if __name__ == "__main__":
    main()
