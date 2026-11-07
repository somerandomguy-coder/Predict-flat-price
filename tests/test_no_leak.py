"""Run with FLATS_MOCK=1. Future rows must not move any feature."""
import numpy as np
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


def _poisoned(L, cutoff):
    L = L.copy()
    L.loc[L.lease_date > cutoff, "rent"] *= 5
    L.loc[L.lease_date > cutoff, "car_space_resolved"] = 1 - L.car_space_resolved
    return L


@pytest.mark.parametrize("fold", ["A", "B"])
def test_index_ignores_rows_after_cutoff(fold):
    from src.index import rent_index
    from src.load import load_leases
    cutoff = C.FOLDS[fold]["cutoff"]
    L = load_leases()
    clean = rent_index(L, cutoff)
    for other in (_poisoned(L, cutoff), L[L.lease_date <= cutoff].copy()):
        i = rent_index(other, cutoff)
        np.testing.assert_array_equal(clean.curves, i.curves)
        np.testing.assert_array_equal(clean.slopes, i.slopes)


def test_fit_index_refuses_rows_after_cutoff():
    from src.index import fit_index
    from src.load import load_leases
    with pytest.raises(AssertionError):
        fit_index(load_leases(), C.FOLDS["B"]["cutoff"])


def test_anchors_ignore_rows_after_cutoff_when_they_are_dropped(monkeypatch):
    from src import features as F
    from src.load import fold_test_rows, load_leases
    cutoff = C.FOLDS["B"]["cutoff"]
    t = fold_test_rows("B").head(3000)
    before = F.build_features(cutoff, t)
    L = load_leases()
    L = L[L.lease_date <= cutoff].reset_index(drop=True)
    monkeypatch.setattr(F, "load_leases", lambda: L)
    pd.testing.assert_frame_equal(before, F.build_features(cutoff, t))


def test_leak_tests_can_fail(monkeypatch):
    """Editing rows the cutoff does allow must move the anchors, or a stale cache could hide a leak."""
    from src import features as F
    from src.load import fold_test_rows, load_leases
    cutoff = C.FOLDS["A"]["cutoff"]
    t = fold_test_rows("A").head(3000)
    before = F.build_features(cutoff, t)
    L = load_leases().copy()
    L.loc[L.lease_date <= cutoff, "rent"] *= 1.5
    monkeypatch.setattr(F, "load_leases", lambda: L)
    after = F.build_features(cutoff, t)
    for c in ("anchor_own", "anchor_bldg", "anchor_block", "best_anchor"):
        assert (after[c] / before[c]).median() == pytest.approx(1.5, rel=0.02), c


def test_a_rows_own_advert_never_feeds_its_anchors():
    """At a cutoff just before an advert, no anchor can be built from that advert."""
    from src import features as F
    from src.load import load_input, load_leases
    L = load_leases()
    late = L[L.gnaf_id.isin(load_input().gnaf_id) & (L.lease_date > "2026-01-01")].drop_duplicates("gnaf_id").head(2000)
    cutoff = late.lease_date.min() - pd.Timedelta(days=1)
    X = F.build_features(cutoff, late.assign(target_date=late.lease_date)[["gnaf_id", "target_date", "car_space_resolved"]])
    n_before = late.gnaf_id.map(L[L.lease_date <= cutoff].groupby("gnaf_id").size()).fillna(0).astype(int).to_numpy()
    np.testing.assert_array_equal(X.anchor_own_n.to_numpy(), n_before)
    assert (X.anchor_own_age_days.dropna() > 0).all()


def test_missing_flags_do_not_reveal_later_dates():
    """A date hidden by the cutoff must not raise the 'unknown' flag."""
    from src import features as F
    from src.load import load_input
    f = load_input()
    cutoff = C.FOLDS["B"]["cutoff"]
    later = f[f.attrs_last_seen > cutoff].head(200)
    assert len(later)
    X = F.build_features(cutoff, later[["gnaf_id"]].assign(target_date=cutoff + pd.Timedelta(days=30), car_space_resolved=1.0))
    assert X.attrs_age_days.isna().all() and (X.attrs_missing == 0).all()


def test_raw_last_rent_never_uses_rows_after_the_cutoff(monkeypatch):
    from src import features as F
    from src.load import fold_test_rows, load_leases
    cutoff = C.FOLDS["A"]["cutoff"]
    t = fold_test_rows("A").head(2000)
    before = F.build_features(cutoff, t)[F.RAW_PREV_COLS]
    L = load_leases().copy()
    L.loc[L.lease_date > cutoff, "rent"] *= 5
    monkeypatch.setattr(F, "load_leases", lambda: L)
    pd.testing.assert_frame_equal(before, F.build_features(cutoff, t)[F.RAW_PREV_COLS])
