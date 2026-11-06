"""Central project paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_PICKLE = RAW_DIR / "LSWMD.pkl"
WAFERS_PARQUET = PROCESSED_DIR / "wafers.parquet"
SPLIT_PARQUET = PROCESSED_DIR / "split.parquet"

FIGURES_DIR = ROOT / "results" / "figures"
METRICS_DIR = ROOT / "results" / "metrics"
CHECKPOINT_DIR = ROOT / "checkpoints"  # model weights (*.pt, git-ignored)

RANDOM_SEED = 42

# Model selection rule: highest val F1 macro among models whose missed-defect rate
# (true defect predicted O.K.) is at most this. A parameter; compare.py tries other values.
MAX_MISSED_DEFECT_RATE = 0.04
