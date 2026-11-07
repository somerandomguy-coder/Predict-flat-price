"""Fold B tests of API data. Fold A is never read here: run_experiment keeps report_fold_a False.

Targets and labels are always the supplied file's, so every variant is scored on the same adverts. "First-time"
is fixed as anchor_own blank in the base panel, so a variant that gives such a flat a history stays in the group.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import api_features as AF
from src import config as C
from src import models as M
from src import predict as P
from src.validate import log_experiment, scored_prediction

OUT = C.OUT_DIR / "api"
SEEDS = (7, 11, 23, 42, 101)
BOOT = 2000
THREADS = dict(num_threads=2)  # the machine ran out of memory with jobs side by side: one job, two threads


def panel(scope: str | None) -> pd.DataFrame:
    path = OUT / f"panel_{scope or 'base'}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    AF.use_history(scope)
    p = P.build_panel()
    AF.use_history(None)
    path.parent.mkdir(parents=True, exist_ok=True)
    p.to_parquet(path)
    return p


def base_cols(p: pd.DataFrame) -> list:
    return list(P.feature_cols())


def _fold_b_errors(p: pd.DataFrame, cols: list, seed: int) -> pd.Series:
    """Absolute errors on Fold B rows, by row label, for the group splits and paired tests."""
    model = M.fit_model(M.train_rows(p, C.FOLDS["B"]["cutoff"], M.N_ORIGINS), cols, "log_ratio", dict(seed=seed, lgb_params=THREADS))
    test = M.fold_rows(p, "B")
    prices = M.compose_prices(M.predict_two_pass(model, test), test["input_car_space"])
    s = scored_prediction(pd.concat([test[["rent", "advert_car_space", "input_car_space"]], prices], axis=1))
    return (s.pred - s.rent).abs(), s.pred


def run(name: str, p: pd.DataFrame, cols: list, first_time: pd.Series, seeds=SEEDS) -> dict:
    """Mean over seeds of the official scores, plus first-time and repeat groups."""
    overall, first, repeat, preds = [], [], [], []
    for seed in seeds:
        r = M.run_experiment(dict(frame=p, name=f"api_{name}_s{seed}", methods=["log_ratio"], feature_cols=cols, seed=seed,
                                  lgb_params=THREADS, log=False, description=name, out_path=OUT / "runs" / f"{name}_s{seed}.json"))
        assert not r["fold_a_reported"] and "A" not in r["folds"]
        err, pred = _fold_b_errors(p, cols, seed)
        assert abs(err.median() - r["folds"]["B"]["mae"]) < 1e-9, "group scores do not reproduce run_experiment"
        overall.append(r["folds"]["B"])
        ft = first_time.reindex(err.index)
        first.append(err[ft].median())
        repeat.append(err[~ft].median())
        preds.append(pred)
    o = pd.DataFrame(overall)
    res = dict(name=name, n_cols=len(cols), foldB_mae=o.mae.mean(), foldB_mae_seed_sd=o.mae.std(), within100=o.within100.mean(),
               r2=o.r2.mean(), bias=o.bias.mean(), first_time_mae=float(np.mean(first)), repeat_mae=float(np.mean(repeat)),
               n=int(o.n.iloc[0]), n_first=int(first_time.reindex(preds[0].index).sum()))
    return res, pd.concat(preds, axis=1).mean(axis=1)


def paired(rent: pd.Series, a: pd.Series, b: pd.Series, mask=None, seed=0) -> dict:
    """Bootstrap of median(|b|) - median(|a|) over adverts, on seed-averaged predictions. Negative means b is better."""
    both = a.index.intersection(b.index)  # API history can make one more advert's parking known
    if mask is not None:
        both = both[mask.reindex(both).to_numpy()]
    ea, eb = (a[both] - rent[both]).abs().to_numpy(), (b[both] - rent[both]).abs().to_numpy()
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(ea), (BOOT, len(ea)))
    d = np.median(eb[i], axis=1) - np.median(ea[i], axis=1)
    return dict(diff=float(np.median(eb) - np.median(ea)), lo=float(np.quantile(d, .025)), hi=float(np.quantile(d, .975)))


def show(results: list) -> None:
    t = pd.DataFrame(results)
    print(t.round(3).to_string(index=False), flush=True)


def history_test() -> None:
    base = panel(None)
    first = base.anchor_own.isna()
    test_idx = M.fold_rows(base, "B").index
    ft = first.loc[test_idx]
    cols = base_cols(base)
    print("base Fold B rows", len(test_idx), "first-time", int(ft.sum()), flush=True)
    results, preds = [], {}
    r, preds["base"] = run("base", base, cols, ft)
    results.append(r)
    for scope in ("input", "building"):
        p = panel(scope)
        assert (p.gnaf_id.astype(str).to_numpy() == base.gnaf_id.astype(str).to_numpy()).all()
        assert (p.target_date.to_numpy() == base.target_date.to_numpy()).all() and (p.rent.to_numpy() == base.rent.to_numpy()).all()
        t = M.fold_rows(p, "B")
        print(scope, "first-time rows with an own anchor now", int(t.anchor_own.notna()[ft].sum()),
              "with a building anchor", int(t.anchor_bldg.notna()[ft].sum()), "base", int(M.fold_rows(base, "B").anchor_bldg.notna()[ft].sum()), flush=True)
        r, preds[scope] = run(f"hist_{scope}", p, cols, ft)
        results.append(r)
    show(results)
    (OUT / "history.json").write_text(json.dumps(results, indent=1))
    for scope in ("input", "building"):
        print(scope, "paired overall", paired(base.rent, preds["base"], preds[scope]), "first-time", paired(base.rent, preds["base"], preds[scope], ft), flush=True)
        e0, e1 = (preds["base"] - base.rent).abs(), (preds[scope] - base.rent).abs()
        print(scope, "largest base errors", e0.nlargest(5).round().tolist(), "same rows now", e1.reindex(e0.nlargest(5).index).round().tolist(), flush=True)
        t, b = M.fold_rows(panel(scope), "B"), M.fold_rows(base, "B")
        print(scope, "median anchor_bldg_n", b.anchor_bldg_n.median(), t.anchor_bldg_n.median(), "mean", b.anchor_bldg_n.mean().round(2), t.anchor_bldg_n.mean().round(2),
              "rows with own anchor", int(b.anchor_own.notna().sum()), int(t.anchor_own.notna().sum()), flush=True)


GROUPS = dict(
    transport=["train_m", "light_rail_m", "bus_m", "ferry_m", "bus_stops_2km"],
    amenities=["supermarket_m", "green_m", "restaurant_m", "supermarkets_1km", "green_1km", "restaurants_1km", "restaurants_300m"],
    schools=["school_m", "public_primary_m", "public_secondary_m", "schools_1km", "private_schools_1km"],
    risks=["heritage_on", "heritage_m", "flood_on", "flood_m", "hv_line_m", "environment_m", "public_housing_share"],
    roads_noise=["road_major_m", "road_any_m", "noise", "noise_max_150m"],
    planning=["height_limit_m", "fsr_limit", "zoning"],
)


def with_api(p: pd.DataFrame, sales: bool) -> pd.DataFrame:
    p = AF.add_market(AF.add_geo(p))
    return AF.add_sales(p) if sales else p


def blank_rates(p: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Blank share in Fold B test rows, training rows and final rows. Final rows are built with this module's joins."""
    fin = P.final_rows().assign(origin=C.AS_OF)
    fin = with_api(fin, "sale_log_price" in cols)
    train = M.train_rows(p, C.FOLDS["B"]["cutoff"], M.N_ORIGINS)
    return pd.DataFrame({"fold_b": M.fold_rows(p, "B")[cols].isna().mean(), "train_b": train[cols].isna().mean(), "final": fin[cols].isna().mean()}).round(4)


def feature_test(sales: bool = False, history: str | None = None) -> None:
    base = panel(None)
    ft = base.anchor_own.isna().loc[M.fold_rows(base, "B").index]
    p = with_api(panel(history), sales)
    cols = base_cols(base)
    all_geo = [c for g in GROUPS.values() for c in g]
    new = all_geo + ["crime_rate"] + AF.MARKET_COLS + (AF.SALE_COLS if sales else [])
    print(blank_rates(p, new).to_string(), flush=True)
    # Most decision-relevant first, and saved after each, so a run cut short shows exactly what finished.
    variants = {"none": [], "all_geo": all_geo, "market": AF.MARKET_COLS, "crime": ["crime_rate"]} | GROUPS
    if sales:
        variants = {"none": [], "sales": AF.SALE_COLS, "geo_sales": all_geo + AF.SALE_COLS}
    path = OUT / f"features_{history or 'file'}_{'sales' if sales else 'nosales'}.json"
    results, preds = [], {}
    for name, extra in variants.items():
        r, preds[name] = run(f"{history or 'file'}_{name}", p, cols + extra, ft, seeds=SEEDS[:3])
        if name != "none":
            r |= dict(paired_overall=paired(p.rent, preds["none"], preds[name]), paired_first=paired(p.rent, preds["none"], preds[name], ft))
        results.append(r | dict(complete=True))
        path.write_text(json.dumps(results, indent=1))
        print(json.dumps(results[-1]), flush=True)
    print("ALL VARIANTS DONE", flush=True)


def geo_test() -> None:
    feature_test()


def leads_test() -> None:
    """All five seeds for the control and the two leads from geo_test. They were picked after seeing Fold B, so this is
    a check on seed noise, not a fresh test."""
    base = panel(None)
    ft = base.anchor_own.isna().loc[M.fold_rows(base, "B").index]
    p = with_api(base, False)
    cols = base_cols(base)
    results, preds = [], {}
    for name, extra in {"none": [], "market": AF.MARKET_COLS, "schools": GROUPS["schools"]}.items():
        r, preds[name] = run(f"lead_{name}", p, cols + extra, ft)
        if name != "none":
            r |= dict(paired_overall=paired(p.rent, preds["none"], preds[name]), paired_first=paired(p.rent, preds["none"], preds[name], ft))
        results.append(r | dict(complete=True))
        (OUT / "leads.json").write_text(json.dumps(results, indent=1))
        print(json.dumps(results[-1]), flush=True)
    print("ALL VARIANTS DONE", flush=True)


# There is deliberately no sales test yet. The sale pull is partial, so coverage would differ between fold rows and
# final rows, the skew already removed once (DECISIONS 20). Add one only when check_api_cache shows sale-history complete.


if __name__ == "__main__":
    globals()[sys.argv[1]]()
