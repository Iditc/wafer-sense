"""Unit tests: map resizing, encoding, augmentation and the lot-based split."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import resize_nearest
from src.data.split import add_lot_type, assign_splits, check_no_leakage
from src.models.dataset import augment, encode


def variants(x: np.ndarray) -> list[np.ndarray]:
    """All 8 rotations and flips of a channels-first array."""
    rotated = [np.rot90(x, k, axes=(1, 2)) for k in range(4)]
    return rotated + [r[:, :, ::-1] for r in rotated]


def test_resize_keeps_size_and_values(small_map):
    out = resize_nearest(small_map, 8)
    assert out.shape == (8, 8)
    assert set(np.unique(out)) <= {0, 1, 2}


def test_resize_to_same_size_is_identity(small_map):
    assert np.array_equal(resize_nearest(small_map, 4), small_map)


def test_resize_up_repeats_each_cell(small_map):
    expected = small_map.repeat(2, axis=0).repeat(2, axis=1)
    assert np.array_equal(resize_nearest(small_map, 8), expected)


def test_resize_handles_non_square_maps():
    wafer = np.array([[0, 1, 2, 1, 0, 2]], dtype=np.uint8).repeat(3, axis=0)
    assert resize_nearest(wafer, 6).shape == (6, 6)


def test_encode_two_channels(small_map):
    x = encode(small_map)
    assert x.shape == (2, 4, 4) and x.dtype == np.float32
    assert np.array_equal(x[0], (small_map > 0).astype(np.float32))
    assert np.array_equal(x[1], (small_map == 2).astype(np.float32))


def test_encode_one_channel_scales_to_unit_range(small_map):
    x = encode(small_map, channels=1)
    assert x.shape == (1, 4, 4)
    assert set(np.unique(x)) == {0.0, 0.5, 1.0}


@pytest.mark.parametrize("seed", range(10))
def test_augment_returns_a_rotation_or_flip(small_map, seed):
    x = encode(small_map)
    out = augment(x, np.random.default_rng(seed))
    assert any(np.array_equal(out, v) for v in variants(x))


def test_augment_reaches_all_eight_variants(small_map):
    x = encode(small_map)
    rng = np.random.default_rng(0)
    seen = {augment(x, rng).tobytes() for _ in range(300)}
    assert len(seen) == 8


def split_meta() -> pd.DataFrame:
    """200 lots of 3 wafers, with two classes and some unlabeled wafers."""
    lots = np.repeat([f"lot{i}" for i in range(200)], 3)
    failure = np.array(["none", "Center", None] * 200, dtype=object)
    return add_lot_type(pd.DataFrame({"lot_name": lots, "failure_type": failure}))


def test_lot_type():
    meta = pd.DataFrame(
        {
            "lot_name": ["a", "a", "b", "b", "c", "c"],
            "failure_type": ["none", "Loc", "none", None, None, None],
        }
    )
    assert add_lot_type(meta)["lot_type"].tolist() == [
        "full",
        "full",
        "mixed",
        "mixed",
        "unlabeled",
        "unlabeled",
    ]


def test_split_keeps_every_lot_whole():
    meta = split_meta()
    meta["split"] = assign_splits(meta)
    assert meta.groupby("lot_name")["split"].nunique().eq(1).all()
    check_no_leakage(meta)


def test_split_shares_are_70_15_15():
    meta = split_meta()
    share = assign_splits(meta).value_counts(normalize=True)
    assert share["train"] == pytest.approx(0.70, abs=0.03)
    assert share["val"] == pytest.approx(0.15, abs=0.03)
    assert share["test"] == pytest.approx(0.15, abs=0.03)


def test_split_is_repeatable():
    assert assign_splits(split_meta()).equals(assign_splits(split_meta()))


def test_leakage_check_fails_on_a_shared_lot():
    meta = pd.DataFrame(
        {"lot_name": ["a", "a", "b"], "split": ["train", "val", "test"]}
    )
    with pytest.raises(AssertionError, match="1 lots"):
        check_no_leakage(meta)
