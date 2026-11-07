"""The index on small synthetic histories with a known trend. No data files needed."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.index import DEFAULT_PARAMS, IndexParams, fit_index

CUTOFF = pd.Timestamp("2026-03-31")
GROWTH = 0.10  # log points a year


def history(n_flats=1500, per_flat=4, link=True, seed=0, drop_months=()):
    rng = np.random.default_rng(seed)
    flat = np.repeat(np.arange(n_flats), per_flat)
    d = CUTOFF - pd.to_timedelta(rng.integers(0, 365 * 6, len(flat)), unit="D")
    years = (d - CUTOFF).days / 365.25
    level = rng.normal(6.4, 0.25, n_flats)[flat]
    h = pd.DataFrame({
        "gnaf_id": pd.Series([f"G{i}" for i in flat], dtype="string"), "lease_date": d,
        "rent": np.exp(level + GROWTH * years + rng.normal(0, 0.04, len(flat))),
        "suburb": np.array(["A", "B", "C"])[flat % 3], "bedrooms": (flat % 3 + 1).astype(float), "car_space_resolved": 1.0})
    if not link:
        h["gnaf_id"] = pd.Series([pd.NA] * len(h), dtype="string")
    return h[~h.lease_date.dt.to_period("M").astype(str).isin(drop_months)].reset_index(drop=True)


def yearly(i, days=730):
    a = i.at([CUTOFF - pd.Timedelta(days=days), CUTOFF])
    return (a[1] - a[0]) / (days / 365.25)


def test_recovers_planted_growth_from_repeats():
    i = fit_index(history(), CUTOFF)
    assert yearly(i) == pytest.approx(GROWTH, abs=0.005)
    assert i.slopes[-1] * 12 == pytest.approx(GROWTH, abs=0.015)


def test_falls_back_to_medians_with_no_repeats():
    i = fit_index(history(link=False), CUTOFF)
    assert i.n_pairs == 0 and yearly(i) == pytest.approx(GROWTH, abs=0.03)


def test_empty_months_are_bridged():
    gap = ["2025-06", "2025-07", "2025-08", "2025-11", "2026-03"]
    i = fit_index(history(drop_months=gap), CUTOFF)
    assert np.isfinite(i.curves).all()
    assert yearly(i) == pytest.approx(GROWTH, abs=0.01)


def test_thin_suburb_falls_back_to_all_suburbs():
    h = history()
    extra = h.head(2).assign(suburb="Tiny", rent=[300.0, 900.0], gnaf_id=pd.array([pd.NA, pd.NA], dtype="string"))
    i = fit_index(pd.concat([h, extra], ignore_index=True), CUTOFF)
    t = i.codes(["Tiny"])[0]
    assert i.slopes[t] == pytest.approx(i.slopes[-1], abs=0.002)


def test_unknown_or_blank_suburb_uses_all_suburbs_row():
    i = fit_index(history(), CUTOFF)
    assert list(i.codes(["Nowhere", None, np.nan, pd.NA])) == [len(i.groups)] * 4


def test_very_short_or_sparse_history_does_not_crash():
    h = history(n_flats=3, per_flat=1)
    h["lease_date"] = CUTOFF - pd.Timedelta(days=5)
    i = fit_index(h, CUTOFF)
    assert np.isfinite(i.at([CUTOFF + pd.Timedelta(days=30)])).all()


def test_zero_at_cutoff_and_rises_after():
    i = fit_index(history(), CUTOFF, IndexParams())
    a = i.at([CUTOFF, CUTOFF + pd.Timedelta(days=61)])
    assert a[0] == pytest.approx(0, abs=1e-9)
    assert a[1] == pytest.approx(GROWTH * 61 / 365.25, abs=0.005)


def test_damping_and_horizon_limit_the_projection():
    h = history()
    later = CUTOFF + pd.Timedelta(days=61)
    full = fit_index(h, CUTOFF, IndexParams()).at([later])[0]
    half = fit_index(h, CUTOFF, replace(IndexParams(), damping=0.8)).at([later])[0]
    flat = fit_index(h, CUTOFF, replace(IndexParams(), damping=0.0)).at([later])[0]
    assert flat == pytest.approx(0, abs=1e-9) and 0 < half < full
    capped = fit_index(h, CUTOFF, replace(IndexParams(), max_horizon_months=1.0))
    a = capped.at([CUTOFF + pd.Timedelta(days=60), CUTOFF + pd.Timedelta(days=300)])
    assert a[0] == pytest.approx(a[1])


def test_flat_index_when_switched_off():
    i = fit_index(history(), CUTOFF, IndexParams(use_index=False))
    assert not i.at([CUTOFF - pd.Timedelta(days=900), CUTOFF + pd.Timedelta(days=60)]).any()
