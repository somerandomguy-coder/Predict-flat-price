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
    model = M.fit_model(M.train_rows(p, C.FOLDS["B"]["cutoff"], M.N_ORIGINS), cols, "log_ratio", dict(seed=seed))
    test = M.fold_rows(p, "B")
    prices = M.compose_prices(M.predict_two_pass(model, test), test["input_car_space"])
    s = scored_prediction(pd.concat([test[["rent", "advert_car_space", "input_car_space"]], prices], axis=1))
    return (s.pred - s.rent).abs(), s.pred


def run(name: str, p: pd.DataFrame, cols: list, first_time: pd.Series, seeds=SEEDS) -> dict:
    """Mean over seeds of the official scores, plus first-time and repeat groups."""
    overall, first, repeat, preds = [], [], [], []
    for seed in seeds:
        r = M.run_experiment(dict(frame=p, name=f"api_{name}_s{seed}", methods=["log_ratio"], feature_cols=cols, seed=seed,
                                  log=False, description=name, out_path=OUT / "runs" / f"{name}_s{seed}.json"))
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
    ea, eb = (a - rent).abs().to_numpy(), (b - rent).abs().to_numpy()
    if mask is not None:
        ea, eb = ea[mask], eb[mask]
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
    rent = base.rent.reindex(preds["base"].index)
    m = ft.reindex(preds["base"].index).to_numpy()
    for scope in ("input", "building"):
        print(scope, "paired overall", paired(rent, preds["base"], preds[scope]), "first-time", paired(rent, preds["base"], preds[scope], m), flush=True)
    (OUT / "history.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    globals()[sys.argv[1]]()
