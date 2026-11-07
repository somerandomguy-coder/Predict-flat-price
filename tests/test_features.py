"""Run with FLATS_MOCK=1. The column contract, and behaviour on thin or messy input."""
import numpy as np
import pandas as pd
import pytest

from src import config as C

pytestmark = pytest.mark.skipif(not C.MOCK, reason="needs FLATS_MOCK=1")
CUTOFF = C.FOLDS["A"]["cutoff"]


def all_flats(park=None):
    from src.load import load_input
    f = load_input()
    return f[["gnaf_id"]].assign(target_date=C.SCORE_MID, car_space_resolved=f.car_space if park is None else park)


@pytest.fixture(scope="module")
def X():
    from src.features import build_features
    return build_features(C.AS_OF, all_flats())


def test_contract_columns(X):
    from src.features import ANCHORS, FEATURE_COLS
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS))
    assert not set(FEATURE_COLS) - set(X.columns)
    for a in ANCHORS:
        assert {a, f"{a}_age_days", f"{a}_n"} <= set(X.columns)


def test_one_row_per_target_in_order(X):
    assert len(X) == C.N_INPUT
    assert X.gnaf_id.equals(all_flats().gnaf_id.reset_index(drop=True))


def test_best_anchor_never_blank_and_most_specific(X):
    assert X.best_anchor.notna().all() and np.isfinite(X.best_anchor).all() and (X.best_anchor > 0).all()
    assert X.anchor_block.notna().all()
    own = X.anchor_own.notna()
    assert (X.best_anchor[own] == X.anchor_own[own]).all()
    bld = ~own & X.anchor_bldg.notna()
    assert (X.best_anchor[bld] == X.anchor_bldg[bld]).all()


def test_genuine_nan_stays_nan(X):
    never = X.ever_advertised_for_lease == 0
    assert never.any() and X.anchor_own[never].isna().all() and (X.anchor_own_n[never] == 0).all()
    assert X.floor_level.isna().any()


def test_suburb_categories_do_not_depend_on_rows():
    from src.features import build_features
    a = build_features(CUTOFF, all_flats().head(50))
    b = build_features(C.FOLDS["B"]["cutoff"], all_flats().tail(50))
    assert list(a.suburb.cat.categories) == list(b.suburb.cat.categories)


def test_target_advert_parking_never_moves_anchors():
    """Anchors use parking known at the cutoff. The target advert's own parking is not known then."""
    from src.features import build_features
    from src.load import fold_test_rows
    t = fold_test_rows("A").head(2000)
    a = build_features(C.FOLDS["A"]["cutoff"], t.assign(car_space_resolved=1.0))
    b = build_features(C.FOLDS["A"]["cutoff"], t.assign(car_space_resolved=0.0))
    for c in ("anchor_own", "anchor_bldg", "anchor_block", "best_anchor"):
        pd.testing.assert_series_equal(a[c], b[c])

def test_bldg_anchor_leaves_out_the_flat_itself(monkeypatch):
    from src import features as F
    from src.load import load_leases
    t = all_flats().head(3000)
    base = F.build_features(CUTOFF, t)
    pick = base[base.anchor_own.notna() & base.anchor_bldg.notna()].gnaf_id.head(300)
    L = load_leases().copy()
    L.loc[L.gnaf_id.isin(pick), "rent"] *= 1.3
    monkeypatch.setattr(F, "load_leases", lambda: L)
    after = F.build_features(CUTOFF, t)
    m = base.gnaf_id.isin(pick)
    assert (after.anchor_own[m] / base.anchor_own[m]).median() == pytest.approx(1.3, rel=0.01)
    assert (after.anchor_bldg[m] / base.anchor_bldg[m]).median() == pytest.approx(1.0, abs=0.01)


def test_target_columns_that_clash_with_flat_attributes(X):
    from src.features import build_features
    t = all_flats().head(100).assign(bedrooms=99, suburb="Elsewhere", rent=1.0)
    out = build_features(C.AS_OF, t)
    assert (out.bedrooms == X.bedrooms.head(100)).all() and "rent" in out


def test_unknown_flat_and_blank_parking_still_get_an_anchor():
    from src.features import build_features
    t = all_flats().head(3).copy()
    t["gnaf_id"] = t.gnaf_id.astype("object")
    t.iloc[0, 0] = "NOT_A_FLAT"
    t["car_space_resolved"] = np.nan
    out = build_features(C.AS_OF, t)
    assert out.best_anchor.notna().all() and out.best_anchor_source[0] == 2


def _thin(monkeypatch, L):
    from src import features as F
    monkeypatch.setattr(F, "load_leases", lambda: L)
    return F


def test_no_flat_link_in_the_leases(monkeypatch):
    from src.load import load_leases
    L = load_leases().copy()
    L["gnaf_id"] = pd.Series([pd.NA] * len(L), dtype="string")
    F = _thin(monkeypatch, L)
    out = F.build_features(CUTOFF, all_flats().head(2000))
    assert out.anchor_own.isna().all() and out.best_anchor.notna().all()


def test_no_building_or_suburb_or_parking_in_the_leases(monkeypatch):
    from src.load import load_leases
    L = load_leases().copy()
    for c in ("building_key", "suburb"):
        L[c] = pd.Series([pd.NA] * len(L), dtype="string")
    L["car_space_resolved"] = np.nan
    L["bedrooms"] = np.nan
    F = _thin(monkeypatch, L)
    out = F.build_features(CUTOFF, all_flats().head(2000))
    assert out.best_anchor.notna().all() and np.isfinite(out.best_anchor).all()


def test_very_few_leases(monkeypatch):
    from src.load import load_leases
    L = load_leases()
    L = L[L.lease_date <= CUTOFF].sample(40, random_state=0).reset_index(drop=True)
    F = _thin(monkeypatch, L)
    out = F.build_features(CUTOFF, all_flats().head(2000))
    assert out.best_anchor.notna().all() and np.isfinite(out.best_anchor).all()
