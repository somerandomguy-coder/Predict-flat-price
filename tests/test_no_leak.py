"""Run with FLATS_MOCK=1. Future rows must not move any feature."""
import pandas as pd
import pytest

from src import config as C

pytestmark = pytest.mark.skipif(not C.MOCK, reason="needs FLATS_MOCK=1")


def test_load_drops_rows_after_as_of():
    from src.load import load_leases
    assert load_leases().lease_date.max() <= C.AS_OF


def test_features_ignore_rows_after_cutoff(monkeypatch):
    from src import features as F
    from src.load import fold_test_rows, load_leases
    cutoff = C.FOLDS["A"]["cutoff"]
    t = fold_test_rows("A").head(3000)
    before = F.build_features(cutoff, t)
    L = load_leases().copy()
    L.loc[L.lease_date > cutoff, "rent"] *= 5
    L.loc[L.lease_date > cutoff, "car_space_resolved"] = 1 - L.car_space_resolved
    monkeypatch.setattr(F, "load_leases", lambda: L)
    pd.testing.assert_frame_equal(before, F.build_features(cutoff, t))


def test_later_dated_input_columns_are_masked():
    from src.features import flats_asof
    cutoff = C.FOLDS["B"]["cutoff"]
    f = flats_asof(cutoff)
    assert f.attrs_last_seen.max() <= cutoff and f.first_ever_listed.max() <= cutoff
