"""Stages 3-4 check: bias before and after the index, and each anchor alone, on Folds A and B.

Projection settings are chosen on Fold B only. Fold A is then reported with them, untouched.
"""
import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from itertools import product

import numpy as np
import pandas as pd

from src import config as C
from src.features import build_features, flats_asof
from src.index import DEFAULT_PARAMS, IndexParams, rent_index
from src.load import fold_test_rows, load_leases
from src.validate import log_experiment, score

NO_INDEX = IndexParams(use_index=False)
GRID = dict(damping=[0.0, 0.8, 0.9, 0.95, 1.0], trend_months=[6, 12, 24])
METHODS = ["baseline", "anchor_own", "anchor_bldg", "anchor_block", "best_anchor"]
AGE_BANDS = [("0-6 months", 0, 183), ("6-12 months", 183, 365), ("1-2 years", 365, 730), ("2-5 years", 730, 1826),
             ("5+ years", 1826, np.inf)]


def scored(X: pd.DataFrame, col: str) -> dict:
    """Score one column alone. Rows the method has no answer for are left out, and n shows how many remain."""
    if col == "baseline":
        pred = X.base_with.where(X.car_space_resolved > 0, X.base_without)
    else:
        pred = X[col]
    Y = X[pred.notna()]
    return score(Y.assign(rent_with_car_space=pred[Y.index], rent_without_car_space=pred[Y.index]))


def show(tag: str, fold: str, col: str, s: dict) -> None:
    print(f"  {fold} {tag:<14}{col:<14} n={s['n']:>6,}  median_abs_err={s['mae']:7.2f}  bias={s['bias']:8.2f}  within100={s['within100']:.3f}")


def best_anchor_age(X: pd.DataFrame) -> pd.Series:
    """Age in days, at target_date, of the data behind best_anchor."""
    ages = X[["anchor_own_age_days", "anchor_bldg_age_days", "anchor_block_age_days"]].to_numpy()
    return pd.Series(ages[np.arange(len(X)), X.best_anchor_source.to_numpy()], index=X.index)


def write_index_bias(before: pd.DataFrame, after: pd.DataFrame) -> None:
    """Fold A bias of best_anchor, in $/week, prediction minus advert, without and with the index."""
    age = best_anchor_age(after)
    by_age = []
    for band, lo, hi in AGE_BANDS:
        m = (age >= lo) & (age < hi)
        if m.any():
            a, b = scored(after[m], "best_anchor"), scored(before[m], "best_anchor")
            by_age.append(dict(age_band=band, n=a["n"], bias_before=round(b["bias"], 2), bias_after=round(a["bias"], 2)))
    out = {"provenance": {"mode": "mock" if C.MOCK else "real", "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
           "fold": "A",
           "overall": {"bias_before": round(scored(before, "best_anchor")["bias"], 2),
                       "bias_after": round(scored(after, "best_anchor")["bias"], 2)},
           "by_age": by_age}
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (C.OUT_DIR / "index_bias.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out))


def tune_on_fold_b() -> IndexParams:
    """Among settings within one bootstrap standard error of the best error, take the lowest |bias|.

    The grid's spread in median error can be smaller than its noise, so the minimum alone would fit noise.
    """
    f, t = C.FOLDS["B"], fold_test_rows("B")
    runs = []
    print("Tuning on Fold B only (best_anchor):")
    for damping, trend in product(GRID["damping"], GRID["trend_months"]):
        p = replace(DEFAULT_PARAMS, damping=damping, trend_months=trend)
        X = build_features(f["cutoff"], t, p)
        s = scored(X, "best_anchor")
        runs.append((p, s, X))
        print(f"  damping={damping:<5} trend_months={trend:<3} median_abs_err={s['mae']:7.2f}  bias={s['bias']:8.2f}")
    _, top, X = min(runs, key=lambda r: r[1]["mae"])
    e = (X.best_anchor - X.rent).abs().to_numpy()
    rng = np.random.default_rng(0)
    se = float(np.std([np.median(rng.choice(e, len(e))) for _ in range(500)]))
    best = min((r for r in runs if r[1]["mae"] <= top["mae"] + se), key=lambda r: abs(r[1]["bias"]))[0]
    print(f"  error noise (1 SE) {se:.2f}; chosen damping={best.damping} trend_months={best.trend_months}")
    return best


def main():
    print(f"DATA: {'MOCK, synthetic, not a result' if C.MOCK else 'REAL'}")
    L = load_leases()
    for k, f in C.FOLDS.items():
        i = rent_index(L, f["cutoff"])
        print(f"Index at Fold {k} cutoff: {i.n_pairs:,} repeat pairs, trend {i.slopes[-1] * 12:.3f} log points a year")

    tuned = tune_on_fold_b()
    res = {}
    for name, params in (("no index", NO_INDEX), ("index", DEFAULT_PARAMS), ("tuned on B", tuned)):
        print(name)
        for k, f in C.FOLDS.items():
            X = build_features(f["cutoff"], fold_test_rows(k), params)
            for col in METHODS:
                res[name, k, col] = scored(X, col)
                show(name, k, col, res[name, k, col])

    a = C.FOLDS["A"]["cutoff"]
    t = fold_test_rows("A")
    write_index_bias(build_features(a, t, NO_INDEX), build_features(a, t, tuned))

    for col in METHODS:
        log_experiment(f"Stage 3-4: {col} alone, index tuned on Fold B", f"{col}", f"damping={tuned.damping},trend={tuned.trend_months}",
                       res["tuned on B", "A", col], res["tuned on B", "B", col])

    flats = flats_asof(C.AS_OF)
    rows = flats[["gnaf_id"]].assign(target_date=C.SCORE_MID, car_space_resolved=flats.car_space)
    t0 = time.perf_counter()
    build_features(C.AS_OF, rows, tuned)
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    X = build_features(C.AS_OF, rows, tuned)
    warm = time.perf_counter() - t0
    print(f"build_features on {len(X):,} rows: {cold:.2f}s first call at a cutoff, {warm:.2f}s after")
    print("best_anchor source (0 own, 1 building, 2 block):", X.best_anchor_source.value_counts(normalize=True).round(3).sort_index().to_dict())
    if C.MOCK:
        tr = pd.read_csv(C.MOCK_TRUTH_CSV, dtype={"gnaf_id": "string"})
        tr["target_date"] = C.SCORE_MID
        tr["car_space_resolved"] = tr.advert_car_space.fillna(tr.input_car_space)
        F = build_features(C.AS_OF, tr[tr.car_space_resolved.notna()], tuned)
        show("frozen", "S", "best_anchor", scored(F, "best_anchor"))


if __name__ == "__main__":
    main()
