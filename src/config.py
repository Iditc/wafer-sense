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

RANDOM_SEED = 42
