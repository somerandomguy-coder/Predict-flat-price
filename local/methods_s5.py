"""Step 5: other model families on the same windows and scorer.

CatBoost: MAE on log(rent / best_anchor), suburb as a native categorical, NaN handled natively.
Linear: median regression of log rent on log anchors and a few flat attributes, fitted by sklearn QuantileRegressor.
Both use the same two passes as LightGBM (car_space_resolved set to 1, then 0).
"""
import sys
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.linear_model import QuantileRegressor

from local import methods_lib as L
from src import models as M

CB_CONFIGS = {
    "cb_d6": dict(depth=6, learning_rate=0.05, iterations=1000),
}
CB_CHECKS = (500, 1000)
CB_SEEDS = (7, 11)


def cb_matrix(df, cols):
    X = df[cols].copy()
    for c in cols:
        if c == "suburb":
            X[c] = X[c].astype(object).where(X[c].notna(), "unknown").astype(str)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    return X


def cb_fit(tr, te, cols, cfg, seed):
    y = np.log(tr.rent.to_numpy(float) / tr.best_anchor.to_numpy(float))
    model = CatBoostRegressor(loss_function="MAE", thread_count=L.THREADS, random_seed=seed, verbose=False, allow_writing_files=False,
                              subsample=0.8, bootstrap_type="Bernoulli", rsm=0.8, **cfg)
    model.fit(Pool(cb_matrix(tr, cols), y, cat_features=["suburb"]))
    anchor = te.best_anchor.to_numpy(float)
    out = {}
    for k in CB_CHECKS:
        pair = []
        for car in (1.0, 0.0):
            X = Pool(cb_matrix(te.assign(car_space_resolved=car), cols), cat_features=["suburb"])
            pair.append(anchor * np.exp(model.predict(X, ntree_end=k)))
        out[k] = tuple(pair)
    return out


def run_catboost(panel, cols):
    for name, cfg in CB_CONFIGS.items():
        fits = {k: {w: [] for w in L.WINDOWS} for k in CB_CHECKS}
        started = time.time()
        for w in L.WINDOWS:
            tr, te = L.split(panel, w)
            for s in CB_SEEDS:
                for k, pair in cb_fit(tr, te, cols, cfg, s).items():
                    fits[k][w].append(pair)
            print(name, w, round(time.time() - started), flush=True)
        for k in CB_CHECKS:
            avg = {w: tuple(np.mean([p[i] for p in fits[k][w]], axis=0) for i in (0, 1)) for w in L.WINDOWS}
            sc = {w: L.score_pair(L.split(panel, w)[1], *avg[w]) for w in L.WINDOWS}
            label = f"s5_{name}_r{k}"
            row = dict(name=label, n_seeds=len(CB_SEEDS), rounds=k, **L.summarise(sc))
            row["run_id"] = L.log_row(label, f"catboost_mae:log_ratio:origins=12:{cfg}", sc["B"], row)
            L.save_preds(label, avg)
            L.append_results("step5", [row])
            print({a: round(b, 3) if isinstance(b, float) else b for a, b in row.items()}, flush=True)


def lin_design(df, car, ref):
    """Design matrix. Blank anchors fall back to the block anchor, with a flag, so every row has a value."""
    base = np.log(df.anchor_block.to_numpy(float))
    own, bldg = np.log(df.anchor_own.to_numpy(float)), np.log(df.anchor_bldg.to_numpy(float))
    has_own, has_bldg = ~np.isnan(own), ~np.isnan(bldg)
    bed = pd.to_numeric(df.bedrooms, errors="coerce").fillna(1).clip(0, 4).to_numpy(float)
    age = np.log1p(df.anchor_own_age_days.fillna(0).to_numpy(float) / 365.25)
    X = {"block": base, "own_gap": np.where(has_own, own - base, 0.0), "bldg_gap": np.where(has_bldg, bldg - base, 0.0),
         "has_own": has_own * 1.0, "has_bldg": has_bldg * 1.0, "own_age": np.where(has_own, age, 0.0),
         "car": np.asarray(car, float), "bath": pd.to_numeric(df.bathrooms, errors="coerce").fillna(1).to_numpy(float)}
    for b in (0, 2, 3, 4):
        X[f"bed{b}"] = (bed == b) * 1.0
    return pd.DataFrame(X, index=df.index)


def run_linear(panel):
    avg = {}
    started = time.time()
    for w in L.WINDOWS:
        tr, te = L.split(panel, w)
        car = tr.car_space_resolved.fillna(tr.car_space_resolved.mean()).gt(0) * 1.0
        ok = tr.rent.notna()
        reg = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs").fit(lin_design(tr[ok], car[ok], None), np.log(tr.rent[ok].to_numpy(float)))
        print("linear", w, dict(zip(lin_design(tr.head(2), [0, 0], None).columns, reg.coef_.round(3))), round(time.time() - started), flush=True)
        avg[w] = tuple(np.exp(reg.predict(lin_design(te, np.full(len(te), c), None))) for c in (1.0, 0.0))
    sc = {w: L.score_pair(L.split(panel, w)[1], *avg[w]) for w in L.WINDOWS}
    row = dict(name="s5_linear_median", n_seeds=1, **L.summarise(sc))
    row["run_id"] = L.log_row("s5_linear_median", "quantile_regression:log_rent:log_anchors:origins=12", sc["B"], row)
    L.save_preds("s5_linear_median", avg)
    L.append_results("step5", [row])
    print({a: round(b, 3) if isinstance(b, float) else b for a, b in row.items()}, flush=True)


def main():
    panel = L.load_panel()
    run_linear(panel)
    run_catboost(panel, L.columns(panel))


if __name__ == "__main__":
    main()
