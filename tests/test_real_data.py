"""Data checks on the real processed files. Run locally with: pytest -m data"""

import pandas as pd
import pytest

from src.config import SPLIT_PARQUET, WAFERS_PARQUET
from src.data.plot_utils import CLASSES
from src.data.split import check_no_leakage

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def meta() -> pd.DataFrame:
    """Every wafer with its lot, label and split."""
    if not (WAFERS_PARQUET.exists() and SPLIT_PARQUET.exists()):
        pytest.skip("processed data not found")
    wafers = pd.read_parquet(
        WAFERS_PARQUET, columns=["wafer_id", "lot_name", "failure_type"]
    )
    split = pd.read_parquet(SPLIT_PARQUET, columns=["wafer_id", "split"])
    return wafers.merge(split, on="wafer_id", how="left", validate="one_to_one")


def test_dataset_size(meta):
    assert len(meta) == 811_457
    assert meta["failure_type"].notna().sum() == 172_950


def test_every_wafer_has_a_split(meta):
    assert meta["split"].isin(["train", "val", "test"]).all()


def test_no_lot_is_shared_between_splits(meta):
    check_no_leakage(meta)


def test_labels_are_known_classes(meta):
    assert set(meta["failure_type"].dropna()) == set(CLASSES)


def test_every_class_is_in_every_split(meta):
    counts = pd.crosstab(meta["failure_type"], meta["split"])
    assert (counts > 0).all(axis=None)
