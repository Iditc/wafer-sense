"""Evaluation shared by all models: F1 macro, per-class report, confusion matrix.

F1 macro is the primary metric: every class counts equally, so a model that
only predicts O.K. scores low even though O.K. is 85% of the data.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.config import METRICS_DIR
from src.data.plot_utils import CLASSES, display, save, setup_style

OK_INDEX = CLASSES.index("none")


def defect_vs_ok(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """The two costly errors, ignoring which defect type was predicted."""
    true_defect, pred_defect = y_true != OK_INDEX, y_pred != OK_INDEX
    return {
        "missed_defect_rate": float((true_defect & ~pred_defect).sum() / max(true_defect.sum(), 1)),
        "false_alarm_rate": float((~true_defect & pred_defect).sum() / max((~true_defect).sum(), 1)),
    }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """All metrics for one set of predictions, as a JSON-ready dict."""
    labels = list(range(len(CLASSES)))
    names = [display(c) for c in CLASSES]
    report = classification_report(y_true, y_pred, labels=labels, target_names=names,
                                   output_dict=True, zero_division=0)
    return {
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        **defect_vs_ok(y_true, y_pred),
        "per_class": {n: {k: round(v, 4) for k, v in report[n].items()} for n in names},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def plot_confusion(cm: np.ndarray, title: str, filename: str) -> None:
    """Row-normalized confusion matrix: each row shows where one true class went."""
    setup_style()
    cm = np.asarray(cm, dtype=float)
    share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    names = [display(c) for c in CLASSES]

    fig, ax = plt.subplots(figsize=(13, 11))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j] > 0:
                color = "white" if share[i, j] > 0.5 else "black"
                ax.text(j, i, f"{share[i, j]:.0%}", ha="center", va="center", color=color, fontsize=16)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    save(fig, filename)


def save_metrics(metrics: dict, name: str) -> None:
    """Write metrics to results/metrics/<name>.json."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / f"{name}.json").write_text(json.dumps(metrics, indent=2))


def summary_line(metrics: dict) -> str:
    """One line for logs: F1 macro, accuracy and the two costly errors."""
    return (f"F1 macro {metrics['f1_macro']:.3f} | accuracy {metrics['accuracy']:.3f} | "
            f"missed defects {metrics['missed_defect_rate']:.1%} | "
            f"false alarms {metrics['false_alarm_rate']:.1%}")
