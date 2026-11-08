"""Run the embedding tests on one trained model (CNN or SimCLR) and save the results.

Embeddings of the labeled train and val wafers are computed once and cached in
data/processed/emb_<name>.npz. Test set stays closed.
Run: python -m src.models.eval_embeddings --model cnn --name cnn_cosine
     python -m src.models.eval_embeddings --model simclr --name simclr_t02
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import CHECKPOINT_DIR, PROCESSED_DIR
from src.models import ae_eval
from src.models import embed_eval as ee
from src.models.cnn import WaferCNN
from src.models.dataset import WaferDataset
from src.models.metrics import OK_INDEX, plot_confusion, save_metrics, summary_line
from src.models.simclr import SimCLR
from src.models.train_utils import get_device


def parse_args() -> argparse.Namespace:
    """Command-line settings."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["cnn", "simclr"], required=True)
    p.add_argument("--name", required=True, help="checkpoint name in checkpoints/")
    p.add_argument("--k", type=int, default=5, help="neighbours for all distance tests")
    p.add_argument("--holdout", default="Scratch", help="defect type treated as new")
    p.add_argument("--false-alarm", type=float, default=0.014, help="O.K. share flagged (the CNN's rate)")
    p.add_argument("--fracs", type=float, nargs="+", default=[0.01, 0.1, 1.0], help="label shares for the probe")
    p.add_argument("--workers", type=int, default=2)
    return p.parse_args()


def load_model(kind: str, name: str, device: torch.device) -> torch.nn.Module:
    """A trained model with an embed() method."""
    model = WaferCNN() if kind == "cnn" else SimCLR()
    model.load_state_dict(torch.load(CHECKPOINT_DIR / f"{name}.pt", map_location=device))
    return model.to(device)


def embeddings(args: argparse.Namespace, device: torch.device) -> dict[str, tuple]:
    """Train and val (embedding, label) pairs, from the cache when it exists."""
    path = PROCESSED_DIR / f"emb_{args.name}.npz"
    if not path.exists():
        model = load_model(args.model, args.name, device)
        out = {}
        for split in ("train", "val"):
            loader = DataLoader(WaferDataset(split), batch_size=512, num_workers=args.workers)
            out[f"{split}_x"], out[f"{split}_y"] = ee.extract(model, loader, device)
            print(f"{split}: {out[f'{split}_x'].shape}")
        np.savez(path, **out)
    data = np.load(path)
    return {s: (data[f"{s}_x"], data[f"{s}_y"]) for s in ("train", "val")}


def main() -> None:
    """Run the four tests, print a summary, save metrics and figures."""
    args = parse_args()
    emb = embeddings(args, get_device())
    train, val = emb["train"], emb["val"]
    y = val[1]
    tag = f"emb_{args.name}"

    similar = ee.similar_cases(train, val, args.k)
    print(f"Similar cases: precision@{args.k} macro {similar['macro']:.3f} | {similar['per_class']}")

    score = ee.anomaly(train, val, args.k)
    anomaly = ae_eval.evaluate_scores(y, {"knn": score}, args.false_alarm, 0.04)
    print("Anomaly (distance to O.K.):", ae_eval.summary_line(anomaly))
    op = anomaly["at_false_alarm"]
    ae_eval.plot_binary_confusion(op["confusion_matrix"],
                                  f"{args.name}: distance to O.K., {op['false_alarm_rate']:.1%} false alarms",
                                  f"{tag}_anomaly_confusion.png")

    new = ee.new_type(train, val, args.holdout, args.k)
    print(f"New type ({args.holdout}): AUROC vs known defects {new['auroc_vs_known_defects']:.3f} | "
          f"flagged as new {new['flagged_as_new']:.1%} at {new['false_alarm']:.0%} false alarms")

    probes = []
    for frac in args.fracs:
        result = ee.probe(train, val, frac)
        probes.append(result)
        print(f"Probe {frac:.0%} labels ({result['n_train']:,} wafers): {summary_line(result)}")
    full = probes[-1]
    plot_confusion(full["confusion_matrix"],
                   f"Linear probe on {args.name}, {full['frac']:.0%} labels: F1 macro {full['f1_macro']:.3f}",
                   f"{tag}_probe_confusion.png")

    save_metrics({"split": "val", "config": vars(args), "similar_cases": similar, "anomaly": anomaly,
                  "new_type": new, "probe": probes,
                  "n_val": int(len(y)), "n_val_defects": int((y != OK_INDEX).sum())}, tag)


if __name__ == "__main__":
    main()
