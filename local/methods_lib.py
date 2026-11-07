"""Harness for the method comparison. Fold B and earlier windows only: the cached panel has no Fold A rows.

Scoring goes through the same path as src.models.evaluate: two passes, compose_prices on input parking, score().
Inner windows are the three two-month windows before Fold B, each trained only on origins before its own cutoff.
"""
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config as C
from src import features as F
from src import models as M
from src.validate import log_experiment, scored_prediction, score

B_CUTOFF = C.FOLDS["B"]["cutoff"]
WINDOWS = {"B": B_CUTOFF, "I1": pd.Timestamp("2026-01-31"), "I2": pd.Timestamp("2025-11-30"), "I3": pd.Timestamp("2025-09-30")}
INNER = ("I1", "I2", "I3")
SEEDS = (7, 11, 23, 42, 101)
THREADS = int(os.environ.get("METHODS_THREADS", 3))
BASE = {**M.LGB_PARAMS, "num_threads": THREADS}
OUT = C.OUT_DIR / "methods"
OUT.mkdir(parents=True, exist_ok=True)
PANEL_PATH = C.OUT_DIR / "methods_panel.parquet"


def load_panel() -> pd.DataFrame:
    p = pd.read_parquet(PANEL_PATH)
    p["origin"], p["target_date"] = pd.to_datetime(p.origin), pd.to_datetime(p.target_date)
    assert p.target_date.max() <= C.FOLDS["B"]["end"], "Fold A rows in the methods panel"
    return p


def columns(panel, drop=()) -> list:
    return M.resolve_features(panel, dict(feature_cols=F.FEATURE_COLS, drop_features=drop))[0]


_cache = {}


def split(panel, window, n=M.N_ORIGINS):
    """Train and test frames for a window, plus the model matrices, cached."""
    key = (id(panel), window, n)
    if key not in _cache:
        cutoff = WINDOWS[window]
        tr, te = M.train_rows(panel, cutoff, n), panel[panel.origin == cutoff]
        assert te.target_date.min() > cutoff and tr.target_date.max() <= cutoff
        _cache[key] = (tr[tr.best_anchor > 0], te)
    return _cache[key]


def lgb_fit(tr, te, cols, method, params=None, rounds=400, seed=7, weight=None, checkpoints=None):
    """Fit once, return {rounds: (pred_with, pred_without)} at each checkpoint. Same maths as M.fit_model."""
    y = np.log(tr.rent.to_numpy(float) / tr.best_anchor.to_numpy(float)) if method == "log_ratio" else tr.rent.to_numpy(float)
    cats = {c: sorted(tr[c].dropna().astype(str).unique()) for c in cols
            if c in M.CATEGORICAL or not pd.api.types.is_numeric_dtype(tr[c])}
    p = {**BASE, "seed": seed, **(params or {})}
    data = lgb.Dataset(M._matrix(tr, cols, cats), y, weight=weight, categorical_feature=list(cats) or "auto")
    booster = lgb.train(p, data, num_boost_round=rounds)
    X = M._matrix(te, cols, cats)
    out = {}
    anchor = te.best_anchor.to_numpy(float)
    for k in checkpoints or [rounds]:
        pair = []
        for car in (1.0, 0.0):
            raw = booster.predict(X.assign(car_space_resolved=car), num_iteration=k)
            pair.append(anchor * np.exp(raw) if method == "log_ratio" else raw)
        out[k] = tuple(pair)
    return out


def prices(te, pw, pwo, premium=None):
    passes = pd.DataFrame({"pred_with": pw, "pred_without": pwo, "premium": pw - pwo}, index=te.index)
    return M.compose_prices(passes, te["input_car_space"], premium)


def frame(te, pw, pwo, premium=None):
    return pd.concat([te[["rent", "advert_car_space", "input_car_space"]], prices(te, pw, pwo, premium)], axis=1)


def score_pair(te, pw, pwo, premium=None) -> dict:
    return score(frame(te, pw, pwo, premium))


def abs_err(te, pw, pwo, premium=None) -> pd.Series:
    s = scored_prediction(frame(te, pw, pwo, premium))
    return (s["pred"] - s["rent"]).abs()


def summarise(per_window: dict) -> dict:
    """per_window: window -> score dict of the ensemble."""
    inner = [per_window[w] for w in INNER if w in per_window]
    row = dict(B_mae=per_window["B"]["mae"], B_w100=per_window["B"]["within100"], B_bias=per_window["B"]["bias"], B_r2=per_window["B"]["r2"])
    if inner:
        row.update(I_mae=float(np.mean([s["mae"] for s in inner])), I_w100=float(np.mean([s["within100"] for s in inner])),
                   **{f"{w}_mae": per_window[w]["mae"] for w in INNER if w in per_window})
    return row


def log_row(name, model, b: dict, row: dict) -> str:
    extra = f"B within100 {b['within100']:.4f}"
    if "I_mae" in row:
        extra += f"; inner mean MAE {row['I_mae']:.2f} within100 {row['I_w100']:.4f}"
    return log_experiment(f"Method lane: {name} ({extra})", "FEATURE_COLS", model, M.NO_SCORE, b)


def save_preds(name, preds: dict) -> None:
    np.savez_compressed(OUT / f"preds_{name}.npz", **{f"{w}_{i}": np.asarray(v) for w, (pw, pwo) in preds.items()
                                                      for i, v in (("w", pw), ("wo", pwo))})


def load_preds(name) -> dict:
    z = np.load(OUT / f"preds_{name}.npz")
    return {w: (z[f"{w}_w"], z[f"{w}_wo"]) for w in WINDOWS if f"{w}_w" in z}


def run_lgb(panel, name, method="log_ratio", params=None, rounds=400, seeds=SEEDS[:3], n=M.N_ORIGINS, weight_fn=None, windows=WINDOWS,
            checkpoints=None, cols=None, log=True, keep=True) -> list:
    """Seed-averaged scores on every window, one result row per checkpoint. weight_fn(train, cutoff) gives row weights."""
    started = time.time()
    cols = cols or columns(panel)
    checks = checkpoints or [rounds]
    per_seed = {k: {w: [] for w in windows} for k in checks}
    for w in windows:
        tr, te = split(panel, w, n)
        weight = weight_fn(tr, WINDOWS[w]) if weight_fn else None
        for s in seeds:
            for k, pair in lgb_fit(tr, te, cols, method, params, max(checks), s, weight, checks).items():
                per_seed[k][w].append(pair)
    rows = []
    for k in checks:
        label = name if len(checks) == 1 else f"{name}_r{k}"
        avg = {w: tuple(np.mean([p[i] for p in per_seed[k][w]], axis=0) for i in (0, 1)) for w in windows}
        scores = {w: score_pair(split(panel, w, n)[1], *avg[w]) for w in windows}
        one = [score_pair(split(panel, "B", n)[1], *p)["mae"] for p in per_seed[k]["B"]]
        row = dict(name=label, method=method, rounds=k, n_origins=n, n_seeds=len(seeds), **summarise(scores),
                   B_seed_mean=float(np.mean(one)), B_seed_sd=float(np.std(one, ddof=1)) if len(one) > 1 else float("nan"))
        if log:
            row["run_id"] = log_row(label, f"lgbm_l1:{method}:origins={n}:rounds={k}:seeds={len(seeds)}:{json.dumps(params or {}, sort_keys=True)}",
                                    scores["B"], row)
        if keep:
            save_preds(label, avg)
        rows.append(row)
        print({a: round(b, 3) if isinstance(b, float) else b for a, b in row.items() if a not in ("method", "run_id")}, flush=True)
    print(f"  {name}: {time.time() - started:.0f}s", flush=True)
    return rows


def append_results(step: str, rows: list) -> None:
    path = OUT / f"{step}.json"
    old = json.loads(path.read_text()) if path.exists() else []
    path.write_text(json.dumps(old + rows, indent=1, default=str))


def paired_bootstrap(err_a: pd.Series, err_b: pd.Series, groups=None, reps=2000, seed=1) -> dict:
    """Median error of b minus a, by resampling adverts, or whole buildings when groups is given. Negative means b is better."""
    rng = np.random.default_rng(seed)
    a, b = err_a.to_numpy(float), err_b.reindex(err_a.index).to_numpy(float)
    diffs = np.empty(reps)
    if groups is None:
        for i in range(reps):
            ix = rng.integers(0, len(a), len(a))
            diffs[i] = np.median(b[ix]) - np.median(a[ix])
    else:
        codes = pd.factorize(groups.reindex(err_a.index))[0]
        members = [np.flatnonzero(codes == g) for g in range(codes.max() + 1)]
        for i in range(reps):
            ix = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
            diffs[i] = np.median(b[ix]) - np.median(a[ix])
    return dict(diff=float(np.median(b) - np.median(a)), lo=float(np.percentile(diffs, 2.5)), hi=float(np.percentile(diffs, 97.5)),
                p_better=float((diffs < 0).mean()))


def bootstrap_ci(err: pd.Series, groups=None, reps=2000, seed=1) -> dict:
    """Interval for one median error and its within-$100 share."""
    rng = np.random.default_rng(seed)
    e = err.to_numpy(float)
    med, w = np.empty(reps), np.empty(reps)
    members = None
    if groups is not None:
        codes = pd.factorize(groups.reindex(err.index))[0]
        members = [np.flatnonzero(codes == g) for g in range(codes.max() + 1)]
    for i in range(reps):
        ix = rng.integers(0, len(e), len(e)) if members is None else \
            np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        med[i], w[i] = np.median(e[ix]), (e[ix] <= 100).mean()
    return dict(mae=float(np.median(e)), mae_lo=float(np.percentile(med, 2.5)), mae_hi=float(np.percentile(med, 97.5)),
                sd=float(med.std(ddof=1)), w100=float((e <= 100).mean()), w100_lo=float(np.percentile(w, 2.5)), w100_hi=float(np.percentile(w, 97.5)))
