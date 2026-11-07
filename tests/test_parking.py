"""Synthetic tests run anywhere. Mock tests need FLATS_MOCK=1 and check recovery of the planted effects."""
import json

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import parking as P

needs_mock = pytest.mark.skipif(not C.MOCK, reason="needs FLATS_MOCK=1")
END = pd.Timestamp("2026-07-31")


def synth(premium=60.0, floor=4.0, proportional=False, n_b=300, per_b=12, seed=0):
    """Adverts with a planted premium. Parked flats have more bathrooms, and a bathroom is worth $150."""
    rng = np.random.default_rng(seed)
    n = n_b * per_b
    b = np.repeat(np.arange(n_b), per_b)
    car = rng.binomial(1, 0.5, n)
    bath = 1 + rng.binomial(1, np.where(car == 1, 0.7, 0.2))
    fl = rng.integers(0, 25, n)
    level = rng.normal(600, 60, n_b)[b]
    flat = np.repeat(np.arange(n), 3)
    t = rng.integers(0, 3650, len(flat))
    date = END - pd.to_timedelta(t, unit="D")
    idx = np.exp(-0.05 * t / 365.25)
    park = premium * (idx if proportional else 1.0)
    rent = (level[flat] + 150 * (bath[flat] - 1)) * idx + floor * fl[flat] + park * car[flat] + rng.normal(0, 25, len(flat))
    return pd.DataFrame({
        "gnaf_id": pd.array([f"f{i}" for i in flat], dtype="string"), "building_key": pd.array([f"B{i}" for i in b[flat]], dtype="string"),
        "suburb": [f"S{i % 5}" for i in b[flat]], "bedrooms": 1.0, "bathrooms": bath[flat].astype(float),
        "car_space_resolved": car[flat].astype(float), "lease_date": date, "rent": rent, "floor_level": fl[flat].astype(float)})


def test_recovers_premium_where_a_raw_gap_is_confounded():
    h = synth()
    raw = h[h.car_space_resolved == 1].rent.mean() - h[h.car_space_resolved == 0].rent.mean()
    assert raw > 110  # bathrooms pass for parking without controls
    fit = P.fit_from_history(h)
    assert abs(fit.table.query("level == 'all'").premium.iloc[0] - 60) < 6
    assert fit.kappa < 0.5


def test_premium_that_moves_with_rents_is_reported_at_cutoff_level():
    fit = P.fit_from_history(synth(premium=100, proportional=True, n_b=600))
    assert abs(fit.table.query("level == 'all'").premium.iloc[0] - 100) < 8


def test_floor_slope_controls_for_parking_and_bathrooms():
    f = P.fit_from_history(synth(floor=4)).floor
    assert abs(f["slope"]["est"] - 4) < 0.5


def test_negative_evidence_is_floored_at_zero():
    t = P.fit_from_history(synth(premium=-50)).table
    assert (t.premium >= 0).all() and t.premium_raw.min() < 0


def test_thin_building_sits_between_its_raw_value_and_its_suburb():
    t = P.fit_from_history(synth(n_b=100)).table
    b = t[t.level == "building"].merge(t.loc[t.level == "suburb", ["suburb", "bedrooms", "premium"]],
                                       on=["suburb", "bedrooms"], suffixes=("", "_sub"))
    b = b[b.premium > 0]
    lo, hi = b[["premium_raw", "premium_sub"]].min(axis=1), b[["premium_raw", "premium_sub"]].max(axis=1)
    assert b.premium.between(lo - 1e-6, hi + 1e-6).all()
    assert b.premium.std() < b.premium_raw.std()


def table():
    rows = [("building", "B1", "S", 1.0, 90.0), ("suburb", None, "S", 1.0, 70.0), ("bedrooms", None, None, 1.0, 60.0),
            ("bedrooms", None, None, 2.0, 80.0), ("all", None, None, np.nan, 50.0)]
    return pd.DataFrame(rows, columns=["level", "building_key", "suburb", "bedrooms", "premium"])


def test_premium_for_walks_the_fallbacks_and_is_never_nan():
    f = pd.DataFrame({"building_key": ["B1", "B2", "B2", "B2", "B1", None], "suburb": ["S", "S", "T", "S", "S", None],
                      "bedrooms": [1, 1, 1, 7, np.nan, 2]}, index=[9, 8, 7, 6, 5, 4])
    s = P.premium_for(f, table())
    assert s.tolist() == [90, 70, 60, 50, 50, 80]
    assert list(s.index) == list(f.index)


def test_no_comparisons_gives_a_zero_premium_not_a_blank():
    h = synth(n_b=5).assign(car_space_resolved=1.0)
    t = P.fit_from_history(h).table
    f = pd.DataFrame({"building_key": ["B0"], "suburb": ["S0"], "bedrooms": [1]})
    assert P.premium_for(f, t).tolist() == [0.0]


def adverts():
    return pd.DataFrame({"rent": [660.0, 500.0, 760.0, 600.0], "rent_with_car_space": [700.0, 700.0, 800.0, 800.0],
                         "rent_without_car_space": [500.0, 500.0, 600.0, 600.0], "advert_car_space": [1, 0, 1, np.nan],
                         "input_car_space": [1, 0, 0, 0]})


def test_true_premium_beats_a_bad_model_gap_on_the_parking_side():
    d = adverts()
    r = P.compare_premium_methods(d, pd.Series(160.0, index=d.index)).set_index(["method", "segment"])
    assert r.loc[("model", "parking"), "mae"] == 40
    assert r.loc[("premium_without", "parking"), "mae"] == 0
    assert r.loc[("premium_without", "parking"), "d_mae"] == -40
    assert r.loc[("premium_without", "no_parking"), "d_mae"] == 0
    assert r.loc[("premium_with", "parking"), "n"] == 2
    assert r.loc[("model", "flipped"), "n"] == 1


def test_mid_anchor_keeps_the_midpoint_and_changes_only_the_gap():
    d = adverts()
    a = P.apply_premium(d, pd.Series(100.0, index=d.index), "mid")
    assert (a.rent_with_car_space - a.rent_without_car_space == 100).all()
    pd.testing.assert_series_equal(a.rent_with_car_space + a.rent_without_car_space, d.rent_with_car_space + d.rent_without_car_space)


def test_empty_segment_does_not_break_the_comparison():
    d = adverts().assign(advert_car_space=[1, 0, 0, np.nan])
    r = P.compare_premium_methods(d, pd.Series(100.0, index=d.index))
    assert (r[r.segment == "flipped"].n == 0).all()


def test_small_cells_lose_every_number():
    c = P._cell("suburb", "S", dict(est=70.0, se=5.0, n=9, n_buildings=9))
    assert c["suppressed"] and c["premium"] is None and c["n"] is None and c["low"] is None
    assert P._cell("suburb", "S", dict(est=70.0, se=5.0, n=50, n_buildings=4))["suppressed"]
    assert not P._cell("suburb", "S", dict(est=70.0, se=5.0, n=50, n_buildings=9))["suppressed"]


# ---------- mock data: planted truth ----------

def mock_truth() -> pd.Series:
    """Planted premium by suburb, by replaying make_mock's first draws (it does not save them).

    The building-to-suburb map in the data is checked against the replay, so drift in make_mock fails here.
    """
    rng = np.random.default_rng(7)
    sizes = np.maximum(1, rng.lognormal(2.6, 1.0, 4200).astype(int))
    rng.permutation(np.resize(np.repeat(np.arange(4200), sizes), C.N_INPUT))
    s_of_b = rng.choice(33, 4200, p=rng.dirichlet(np.ones(33) * 3))
    rng.uniform(520, 880, 33)
    prem = rng.uniform(40, 110, 33)
    f = pd.read_csv(C.INPUT_CSV, usecols=["building_address", "suburb"])
    f["s"] = s_of_b[f.building_address.str.extract(r"#(\d+)$")[0].astype(int)]
    assert f.groupby("suburb").s.nunique().max() == 1 and f.groupby("s").suburb.nunique().max() == 1, "make_mock changed"
    return f.groupby("suburb").s.first().map(lambda i: prem[i])


@pytest.fixture(scope="module")
def fit():
    return P.fit_premium(C.AS_OF)


@needs_mock
def test_mock_suburb_premiums_are_recovered(fit):
    truth = mock_truth()
    assert truth.between(40, 110).all()
    d = pd.DataFrame(P.parking_report(fit)["by_suburb"]).set_index("suburb")
    err = d.premium - truth.reindex(d.index)
    assert err.abs().mean() < 7 and abs(err.mean()) < 3 and err.abs().max() < 30
    assert (d.low.le(truth) & d.high.ge(truth)).mean() >= 0.8


@needs_mock
def test_mock_premium_is_fixed_in_dollars(fit):
    assert fit.kappa < 0.3


@needs_mock
def test_mock_shrunk_cells_beat_raw_cells(fit):
    truth = mock_truth()
    b = fit.table[fit.table.level == "building"]
    t = b.suburb.map(truth)
    assert (b.premium - t).abs().mean() < 0.3 * (b.premium_raw - t).abs().mean()
    assert abs((b.premium - t).mean()) < 5


@needs_mock
def test_mock_floor_slope_and_curve(fit):
    rep = P.floor_report(fit)
    assert abs(rep["per_floor_premium"] - 4) < 0.5
    curve = pd.DataFrame(rep["curve"])
    assert curve.premium.is_monotonic_increasing and (curve.n > 0).all()
    assert abs(np.polyfit(curve.floor, curve.premium, 1)[0] - 4) < 0.6
    assert rep["reference_floor"] == curve.floor.iloc[0] and curve.premium.iloc[0] == 0


@needs_mock
def test_every_input_flat_gets_a_premium(fit):
    from src.load import load_input
    s = P.premium_for(load_input(), fit.table)
    assert s.notna().all() and (s >= 0).all() and s.between(0, 300).mean() > 0.99


@needs_mock
def test_table_ignores_rows_after_cutoff(monkeypatch):
    from src import features as F
    from src.load import load_leases
    cutoff = C.FOLDS["A"]["cutoff"]
    before = P.estimate_premium(cutoff)
    L = load_leases().copy()
    L.loc[L.lease_date > cutoff, "rent"] *= 5
    L.loc[L.lease_date > cutoff, "car_space_resolved"] = 1 - L.car_space_resolved
    monkeypatch.setattr(F, "load_leases", lambda: L)
    pd.testing.assert_frame_equal(before, P.estimate_premium(cutoff))


@needs_mock
def test_fold_premium_covers_every_test_advert():
    from src.load import fold_test_rows
    t = fold_test_rows("B").head(500)
    assert P.fold_premium("B", t).notna().all()


@needs_mock
def test_reports_hold_aggregates_only(fit, tmp_path, monkeypatch):
    monkeypatch.setattr(C, "OUT_DIR", tmp_path)
    paths = P.write_reports(fit)
    text = "".join(p.read_text() for p in paths.values())
    assert not any(k in text for k in fit.table.building_key.dropna().head(200))
    assert "GANSW" not in text


@needs_mock
def test_reports_match_the_shape_the_report_reads(fit, tmp_path, monkeypatch):
    monkeypatch.setattr(C, "OUT_DIR", tmp_path)
    paths = P.write_reports(fit)
    park, floor = (json.loads(paths[n].read_text()) for n in ("parking.json", "floor.json"))
    for rep in (park, floor):
        assert rep["provenance"]["mode"] == "mock"
        assert pd.Timestamp(rep["provenance"]["generated_at"]).tzinfo is not None
    cell = {"premium", "low", "high", "n"}
    assert cell <= set(park["overall"])
    assert cell | {"suburb"} <= set(park["by_suburb"][0])
    assert cell | {"bedrooms"} <= set(park["by_bedrooms"][0])
    assert isinstance(floor["per_floor_premium"], float) and isinstance(floor["reference_floor"], int)
    assert cell | {"floor"} <= set(floor["curve"][0])
