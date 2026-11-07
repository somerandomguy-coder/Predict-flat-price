"""Model lane. A small synthetic panel keeps most of these independent of the raw CSVs."""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import models as M

FAST = dict(num_boost_round=40, lgb_params=dict(min_data_in_leaf=5, num_leaves=15), n_origins=3)


def make_panel(n=300, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for origin in M.origins_for(C.AS_OF, 5):
        car = rng.integers(0, 2, n).astype(float)
        beds = rng.integers(0, 4, n).astype(float)
        base = 500 + 200 * beds
        rent = base * np.exp(rng.normal(0, .02, n)) + 80 * car
        parts.append(pd.DataFrame({
            "gnaf_id": [f"G{i}" for i in range(n)], "origin": origin, "rent": rent,
            "target_date": origin + pd.to_timedelta(rng.integers(1, 59, n), unit="D"),
            "advert_car_space": car, "input_car_space": np.where(rng.random(n) < .1, np.nan, car), "car_space_resolved": car,
            "car_spaces": car, "bedrooms": beds, "suburb": rng.choice(["A (NSW)", "B (NSW)"], n),
            "floor_level": np.where(rng.random(n) < .4, np.nan, rng.integers(0, 30, n)),
            "base_with": base + 80, "base_without": base, "building_key": "X", "mb": "1",
            "data_mode": "mock" if C.MOCK else "real"}))
    return pd.concat(parts, ignore_index=True)


@pytest.fixture
def no_log(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "EXPERIMENTS_CSV", tmp_path / "experiments.csv")
    monkeypatch.setattr(C, "OUT_DIR", tmp_path)
    return tmp_path


def test_models_never_imports_the_raw_data_path():
    code = ("import sys, src.models; bad = {'src.load', 'src.features', 'src.predict'} & set(sys.modules); "
            "assert not bad, bad; assert callable(src.models.run_experiment)")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=C.ROOT)


def test_run_experiment_from_parquet_alone(tmp_path, no_log, monkeypatch):
    path = tmp_path / "panel.parquet"
    make_panel().to_parquet(path)
    monkeypatch.setattr(C, "INPUT_CSV", tmp_path / "missing.csv")
    monkeypatch.setattr(C, "LEASES_CSV", tmp_path / "missing.csv")
    r = M.run_experiment({**FAST, "features_path": path, "name": "abl"})
    assert r["winner"] in M.METHODS and set(r["method_comparison"]["B"]) == {*M.METHODS, "baseline_hier_median"}
    assert {"mae", "within100", "r2", "bias"} <= set(r["folds"]["B"])
    assert (tmp_path / "ablations" / "abl.json").exists()
    assert (tmp_path / "experiments.csv").read_text().count("\n") == 2  # header and one run


def test_ablation_never_reads_fold_a(no_log):
    p = make_panel()
    r = M.run_experiment({**FAST, "frame": p})
    assert "A" not in r["folds"] and "baseline_fold_a" not in r and r["fold_a_reported"] is False


def test_fold_a_cannot_change_the_choice_or_fold_b(no_log):
    p = make_panel()
    clean = M.run_experiment({**FAST, "frame": p, "report_fold_a": True})
    q = p.copy()
    q.loc[q.origin == C.FOLDS["A"]["cutoff"], "rent"] *= 7
    dirty = M.run_experiment({**FAST, "frame": q, "report_fold_a": True})
    assert clean["winner"] == dirty["winner"]
    assert clean["method_comparison"]["B"] == dirty["method_comparison"]["B"]
    assert clean["folds"]["A"]["mae"] != dirty["folds"]["A"]["mae"]


def test_only_the_winner_reaches_fold_a(no_log, monkeypatch):
    calls, fit = [], M.fit_model
    monkeypatch.setattr(M, "fit_model", lambda t, c, m, cfg=None: calls.append((t.target_date.max(), m)) or fit(t, c, m, cfg))
    r = M.run_experiment({**FAST, "frame": make_panel(), "report_fold_a": True})
    late = [m for d, m in calls if d > C.FOLDS["B"]["cutoff"]]
    assert late == [r["winner"]] and len(calls) == len(M.METHODS) + 1


def test_winner_is_median_error_then_within_100():
    assert M.pick_winner(dict(a=dict(mae=10, within100=.1), b=dict(mae=9, within100=.0))) == "b"
    assert M.pick_winner(dict(a=dict(mae=9, within100=.1), b=dict(mae=9, within100=.2))) == "b"


def test_origins_are_month_ends_two_months_apart():
    assert [str(o.date()) for o in M.origins_for("2026-07-31", 4)] == ["2026-05-31", "2026-03-31", "2026-01-31", "2025-11-30"]
    assert str(M.origins_for("2025-11-30", 2)[-1].date()) == "2025-07-31"  # not 07-30


@pytest.mark.parametrize("fold", ["A", "B"])
def test_training_labels_stop_at_the_fold_cutoff(fold):
    cutoff = C.FOLDS[fold]["cutoff"]
    t = M.train_rows(make_panel(), cutoff, 3)
    assert t.target_date.max() <= cutoff and t.origin.max() < cutoff and t.origin.nunique() == 3


def test_train_rows_rejects_a_late_label():
    p = make_panel()
    cutoff = C.FOLDS["A"]["cutoff"]
    p.loc[p.origin == M.origins_for(cutoff, 1)[0], "target_date"] = cutoff + pd.Timedelta(days=1)
    with pytest.raises(AssertionError):
        M.train_rows(p, cutoff, 3)


def test_shim_excludes_labels_ids_and_parking_proxies():
    cols, source = M.resolve_features(M.ensure_best_anchor(make_panel()))
    assert source == "available_columns_shim"
    assert "base_suburb" not in M.resolve_features(make_panel().assign(base_suburb=1.0, best_anchor=1.0))[0]
    assert not {"rent", "advert_car_space", "input_car_space", "car_spaces", "gnaf_id", "origin", "mb", "building_key"} & set(cols)
    assert {"suburb", "car_space_resolved", "floor_level", "best_anchor"} <= set(cols)


def test_feature_cols_switch_the_shim_off():
    p = make_panel()
    p.attrs["feature_cols"] = ["bedrooms", "suburb", "car_spaces", "car_space_resolved"]
    cols, source = M.resolve_features(p)
    assert source == "FEATURE_COLS" and cols == ["bedrooms", "suburb", "car_space_resolved"]


def test_rent_in_a_feature_list_is_refused():
    with pytest.raises(ValueError):
        M.resolve_features(make_panel(), dict(feature_cols=["rent", "bedrooms"]))


def test_best_anchor_shim_steps_aside():
    p = make_panel()
    assert (M.ensure_best_anchor(p).best_anchor == p.base_without).all()
    q = p.assign(best_anchor=1.0)
    assert (M.ensure_best_anchor(q).best_anchor == 1.0).all()


def test_nan_stays_nan_and_new_suburb_becomes_nan():
    p = M.ensure_best_anchor(make_panel())
    cats = dict(suburb=["A (NSW)"])
    X = M._matrix(p, ["floor_level", "suburb"], cats)
    assert X.floor_level.isna().sum() == p.floor_level.isna().sum() > 0
    assert X.suburb.dtype.name == "category" and X.suburb.isna().sum() == (p.suburb != "A (NSW)").sum()


@pytest.mark.parametrize("method", M.METHODS)
def test_two_pass_recovers_a_planted_premium(method):
    p = M.ensure_best_anchor(make_panel())
    cols, _ = M.resolve_features(p)
    m = M.fit_model(p, cols, method, dict(num_boost_round=150, lgb_params=dict(min_data_in_leaf=5, num_leaves=15)))
    t = M.predict_two_pass(m, p)
    assert list(t.columns) == ["pred_with", "pred_without", "premium"]
    assert 60 < t.premium.median() < 100  # planted at $80
    assert np.allclose(t.pred_with - t.pred_without, t.premium)


def passes(w, wo):
    return pd.DataFrame({"pred_with": w, "pred_without": wo, "premium": np.subtract(w, wo)}).astype(float)


def test_known_parking_keeps_its_own_pass_and_unknown_keeps_both():
    out = M.compose_prices(passes([700, 700, 700], [600, 600, 600]), [1, 0, np.nan], premium=[50, 50, 50])
    assert list(out.rent_with_car_space) == [700, 650, 700]
    assert list(out.rent_without_car_space) == [650, 600, 600]


def test_premium_is_floored_at_zero_from_either_source():
    out = M.compose_prices(passes([600, 600], [700, 700]), [1, 0])
    assert list(out.rent_with_car_space) == [600, 700] and list(out.rent_without_car_space) == [600, 700]
    out = M.compose_prices(passes([700], [600]), [1], premium=[-40])
    assert list(out.rent_without_car_space) == [700]


def test_table_gap_falls_back_to_two_pass():
    out = M.compose_prices(passes([700], [620]), [1], premium=[np.nan])
    assert list(out.rent_without_car_space) == [620]


def test_composed_prices_stay_positive_and_ordered():
    out = M.compose_prices(passes([150, 400], [420, 150]), [1, np.nan], premium=[500, 500])
    assert (out.rent_without_car_space >= M.PRED_FLOOR).all()
    assert (out.rent_with_car_space >= out.rent_without_car_space).all()


def test_blank_prediction_stays_blank_for_the_fallback():
    out = M.compose_prices(passes([np.nan], [np.nan]), [1])
    assert out.isna().all(axis=None)


def test_panel_of_the_other_mode_is_refused(no_log):
    p = make_panel().assign(data_mode="real" if C.MOCK else "mock")
    with pytest.raises(ValueError):
        M.run_experiment({**FAST, "frame": p})


def test_saved_model_predicts_the_same(tmp_path):
    p = M.ensure_best_anchor(make_panel())
    cols, _ = M.resolve_features(p)
    m = M.fit_model(p, cols, "log_ratio", FAST)
    m.save(tmp_path / "model")
    assert np.allclose(m.predict(p), M.Model.load(tmp_path / "model").predict(p))


def test_shap_summary_holds_aggregates_only():
    p = M.ensure_best_anchor(make_panel())
    cols, _ = M.resolve_features(p)
    s = M.shap_summary(M.fit_model(p, cols, "direct", FAST), p, n=100)
    assert len(s["top_features"]) == len(cols) and s["n_rows"] == 100
    assert set(s["top_features"][0]) == {"feature", "label", "mean_abs_shap"}
    assert s["top_features"][0]["feature"] == "bedrooms" and s["unit"] == "$/week"
    assert s["provenance"]["mode"] == ("mock" if C.MOCK else "real")
    json.dumps(s)


def test_shap_on_the_ratio_model_does_not_claim_dollars():
    p = M.ensure_best_anchor(make_panel())
    cols, _ = M.resolve_features(p)
    assert M.shap_summary(M.fit_model(p, cols, "log_ratio", FAST), p, n=50)["unit"] != "$/week"


def test_metrics_match_the_report_shape(no_log):
    p = make_panel().assign(base_suburb=700.0)
    r = M.run_experiment({**FAST, "frame": p, "report_fold_a": True})
    assert r["provenance"]["mode"] == ("mock" if C.MOCK else "real") and "generated_at" in r["provenance"]
    assert r["sources"] == {"nsw_bond_used": False} and isinstance(r["method_summary"], str)
    by = {m["key"]: m for m in r["methods"]}
    assert {"suburb_median", "baseline", "final"} <= set(by)
    for key in ("suburb_median", "baseline", "final"):
        assert set(by[key]["folds"]) == {"A", "B"} and {"n", "mae", "within100", "r2", "bias"} <= set(by[key]["folds"]["B"])
    loser = next(m for m in M.METHODS if m != r["winner"])
    assert set(by[loser]["folds"]) == {"B"}  # the loser is never scored on Fold A


def test_ablation_methods_hold_no_fold_a(no_log):
    r = M.run_experiment({**FAST, "frame": make_panel()})
    assert all(set(m["folds"]) == {"B"} for m in r["methods"])




def test_shim_switches_off_when_feature_cols_exists(monkeypatch):
    from src import features as F
    from src.predict import feature_cols
    assert feature_cols() is getattr(F, "FEATURE_COLS", None)
    monkeypatch.setattr(F, "FEATURE_COLS", ["bedrooms", "suburb"], raising=False)
    p = make_panel()
    p.attrs["feature_cols"] = feature_cols()
    assert M.resolve_features(p)[1] == "FEATURE_COLS"


mock_only = pytest.mark.skipif(not C.MOCK, reason="needs FLATS_MOCK=1")


@pytest.fixture(scope="module")
def real_panel():
    from src.predict import build_panel
    return build_panel(2)


@mock_only
def test_panel_rows_sit_within_two_months_of_their_origin(real_panel):
    p = real_panel
    assert p.origin.nunique() == 4 and p.origin.max() == C.FOLDS["A"]["cutoff"]
    assert (p.target_date > p.origin).all()
    assert (p.target_date <= p.origin + pd.offsets.MonthEnd(2)).all()
    assert p.target_date.max() <= C.AS_OF


@mock_only
@pytest.mark.parametrize("fold", ["A", "B"])
def test_panel_test_rows_match_the_official_fold(real_panel, fold):
    from src.load import fold_test_rows
    key = ["gnaf_id", "target_date", "rent"]
    got = M.fold_rows(real_panel, fold)[key].sort_values(key).reset_index(drop=True)
    want = fold_test_rows(fold)[key].sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(got, want, check_dtype=False)


@mock_only
def test_rents_after_an_origin_move_only_that_rows_label(monkeypatch):
    from src import features as F
    from src import predict as P
    from src.load import load_leases
    origin = M.origins_for(C.FOLDS["A"]["cutoff"], 1)[0]

    def rows_at_origin():
        return M.train_rows(P.build_panel(2), C.FOLDS["A"]["cutoff"], 1)

    before = rows_at_origin()
    L = load_leases().copy()
    L.loc[L.lease_date > origin, "rent"] *= 5
    monkeypatch.setattr(F, "load_leases", lambda: L)
    monkeypatch.setattr(P, "load_leases", lambda: L)
    after = rows_at_origin()
    assert (after.rent == before.rent * 5).all()
    pd.testing.assert_frame_equal(before.drop(columns="rent"), after.drop(columns="rent"))


@mock_only
def test_panel_survives_parquet_with_its_feature_list(real_panel, tmp_path):
    p = real_panel.copy()
    p.attrs["feature_cols"] = ["bedrooms", "suburb"]
    p.to_parquet(tmp_path / "p.parquet")
    q = M.load_panel(dict(features_path=tmp_path / "p.parquet"))
    assert q.attrs["feature_cols"] == ["bedrooms", "suburb"] and len(q) == len(p)
    assert "best_anchor" in q.columns
