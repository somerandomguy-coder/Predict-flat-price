"""Foundation-model lane: one harness so every model sees the same rows, features and scoring.

Fold B only. The panel is read with a parquet filter on origin, so Fold A rows never reach Python.
Local inference only: nothing here calls a hosted model, and telemetry is switched off before imports.

  run    fit models and save test predictions (row-level, so they stay in FM_RESULTS, outside the repo)
  score  turn saved predictions into seed-averaged scores and paired bootstrap gaps

Use a separate venv with CPU-only torch, on disk rather than /tmp, which is RAM here. Run one job at a time:
  FM_RESULTS=/path/res PYTHONPATH=. /path/venv/bin/python local/run_foundation.py run \
      --models lgb,cat,xgb --targets log_ratio,rent --n 0 --seeds 1,2,3 --threads 2
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src import config as C
from src import models as M
from src.validate import log_experiment, score, scored_prediction

PANEL = Path(os.environ.get("FM_PANEL", "/home/nam/Documents/git-repos/active/Predict-flat-price/out/real/features_panel.parquet"))
RES = Path(os.environ.get("FM_RESULTS", "fm_results"))
CUTOFF = C.FOLDS["B"]["cutoff"]
DEVICE = os.environ.get("FM_DEVICE", "cpu")
VAL_ORIGIN = pd.Timestamp("2026-01-31")  # last training origin; its labels end at the Fold B cutoff


def load_panel() -> tuple[pd.DataFrame, list[str]]:
    """Rows with origin up to the Fold B cutoff. Fold A's origin is later, so its rows are filtered in the reader."""
    meta = pq.ParquetFile(PANEL).schema_arrow.metadata
    cols = json.loads(meta[b"PANDAS_ATTRS"])["feature_cols"]
    p = pd.read_parquet(PANEL, filters=[("origin", "<=", CUTOFF)])
    p = p.assign(origin=pd.to_datetime(p.origin), target_date=pd.to_datetime(p.target_date))
    assert p.origin.max() == CUTOFF and p.target_date.max() <= C.FOLDS["B"]["end"], "rows beyond Fold B were loaded"
    assert set(p.data_mode.unique()) == {"real"}
    p.attrs["feature_cols"] = cols
    return p, M.resolve_features(p)[0]


def frames(panel: pd.DataFrame, val: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, test). val=True holds out the last training origin, for choosing blends before Fold B is scored."""
    train = M.train_rows(panel, CUTOFF, M.N_ORIGINS)
    train = train[train.best_anchor > 0]
    if val:
        return train[train.origin < VAL_ORIGIN], train[train.origin == VAL_ORIGIN]
    return train, M.fold_rows(panel, "B")


def subsample(train: pd.DataFrame, n: int, seed: int, how: str = "random") -> pd.DataFrame:
    """The training rows for this seed. Every model given the same n, seed and how gets the same rows."""
    if not n or n >= len(train):
        return train
    if how == "recent":
        return train.sort_values("target_date", kind="stable").tail(n)
    return train.iloc[np.sort(np.random.default_rng(1000 + seed).choice(len(train), n, replace=False))]


def design(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Model input in the shape src.models uses, with test stacked as car=1 rows then car=0 rows."""
    cats = {c: sorted(train[c].dropna().astype(str).unique()) for c in cols
            if c in M.CATEGORICAL or not pd.api.types.is_numeric_dtype(train[c])}
    Xtr, Xte = M._matrix(train, cols, cats), M._matrix(test, cols, cats)
    both = pd.concat([Xte.assign(car_space_resolved=1.0), Xte.assign(car_space_resolved=0.0)], ignore_index=True)
    return Xtr, both, list(cats)


def ordinal(Xtr: pd.DataFrame, Xte: pd.DataFrame, cats: list[str]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Float arrays for models that take a plain matrix. Category codes, with NaN for blank."""
    def enc(X):
        X = X.copy()
        for c in cats:
            X[c] = X[c].cat.codes.astype(float).replace(-1, np.nan)
        return X.to_numpy(np.float32)
    return enc(Xtr), enc(Xte), [Xtr.columns.get_loc(c) for c in cats]


def impute(Xtr: pd.DataFrame, Xte: pd.DataFrame, cats: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For models that refuse NaN: training medians plus a flag per blank column. Blank categories become their own level."""
    med = Xtr.drop(columns=cats).median()
    flags = [c for c in med.index if Xtr[c].isna().any()]

    def fill(X):
        out = X.assign(**{f"{c}_blank": X[c].isna().astype(float) for c in flags}).fillna(med)
        for c in cats:
            out[c] = out[c].cat.add_categories("NA").fillna("NA")
        return out
    return fill(Xtr), fill(Xte)


def target(train: pd.DataFrame, form: str) -> np.ndarray:
    rent = train.rent.to_numpy(float)
    return np.log(rent / train.best_anchor.to_numpy(float)) if form == "log_ratio" else rent


def to_rent(raw: np.ndarray, test: pd.DataFrame, form: str) -> tuple[np.ndarray, np.ndarray]:
    """Split the stacked predictions into (pred_with, pred_without) in weekly dollars."""
    w, wo = raw[:len(test)], raw[len(test):]
    if form == "log_ratio":
        a = test.best_anchor.to_numpy(float)
        return a * np.exp(w), a * np.exp(wo)
    return w, wo


# ---- models: fit on (Xtr, y), return {variant: stacked predictions} ----------------------------------------

def m_lgb(Xtr, y, Xte, cats, seed, threads, **_):
    import lightgbm as lgb
    params = {**M.LGB_PARAMS, "seed": seed, "num_threads": threads}
    b = lgb.train(params, lgb.Dataset(Xtr, y, categorical_feature=cats), num_boost_round=M.NUM_BOOST_ROUND)
    return {"": b.predict(Xte)}


def m_cat(Xtr, y, Xte, cats, seed, threads, **_):
    from catboost import CatBoostRegressor

    def prep(X):
        X = X.copy()
        for c in cats:
            X[c] = X[c].astype(object).where(X[c].notna(), "NA").astype(str)
        return X
    m = CatBoostRegressor(loss_function="MAE", iterations=800, learning_rate=0.05, depth=6, random_seed=seed,
                          thread_count=threads, verbose=False, allow_writing_files=False, cat_features=cats)
    m.fit(prep(Xtr), y)
    return {"": m.predict(prep(Xte))}


def m_xgb(Xtr, y, Xte, cats, seed, threads, **_):
    import xgboost as xgb
    params = dict(objective="reg:absoluteerror", tree_method="hist", eta=0.05, max_depth=7, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=5, seed=seed, nthread=threads, verbosity=0)
    b = xgb.train(params, xgb.DMatrix(Xtr, y, enable_categorical=True), num_boost_round=M.NUM_BOOST_ROUND)
    return {"": b.predict(xgb.DMatrix(Xte, enable_categorical=True))}


def m_knn(Xtr, y, Xte, cats, seed, threads, **_):
    """Comparables: the median target of the k most similar flats, on a hand-weighted distance."""
    from sklearn.neighbors import NearestNeighbors

    def geo(X):
        anchor = np.log(X["best_anchor"].to_numpy(float).clip(100))
        return np.c_[X["latitude"].fillna(X["latitude"].median()) * 111.0 / 0.5,   # about 500 m per unit
                     X["longitude"].fillna(X["longitude"].median()) * 92.0 / 0.5,
                     X["bedrooms"].fillna(1) * 3, X["bathrooms"].fillna(1) * 1.5, X["car_space_resolved"].fillna(0) * 2,
                     anchor * 8, X["best_anchor_source"].fillna(0) * 2]
    nn = NearestNeighbors(n_neighbors=50, n_jobs=threads).fit(geo(Xtr))
    idx = nn.kneighbors(geo(Xte), return_distance=False)
    return {"k50": np.median(y[idx], axis=1), "k15": np.median(y[idx[:, :15]], axis=1)}


def m_tabpfn(Xtr, y, Xte, cats, seed, threads, n_estimators=2, version="v2", **_):
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    torch.set_num_threads(threads)
    A, B, ci = ordinal(Xtr, Xte, cats)
    make = TabPFNRegressor.create_default_for_version
    m = make(ModelVersion.V2, n_estimators=n_estimators, device=DEVICE, random_state=seed,
             ignore_pretraining_limits=True, categorical_features_indices=ci, memory_saving_mode=True)
    m.fit(A, y)
    p = m.predict(B, output_type="main")
    return {"mean": p["mean"], "median": p["median"]}


def m_tabicl(Xtr, y, Xte, cats, seed, threads, n_estimators=4, **_):
    import torch
    from tabicl import TabICLRegressor
    torch.set_num_threads(threads)
    A, B, _ci = ordinal(Xtr, Xte, cats)
    m = TabICLRegressor(n_estimators=n_estimators, device=DEVICE, random_state=seed, n_jobs=threads)
    m.fit(A, y)
    p = m.predict(B, output_type=["mean", "median"])
    return {"mean": np.asarray(p["mean"]), "median": np.asarray(p["median"])}


def m_realmlp(Xtr, y, Xte, cats, seed, threads, **_):
    import torch
    from pytabkit import RealMLP_TD_Regressor
    torch.set_num_threads(threads)
    kw = dict(device=DEVICE, random_state=seed, n_cv=1, n_refit=0, verbosity=0, n_threads=threads)
    m = RealMLP_TD_Regressor(**kw, n_epochs=int(os.environ.get("FM_EPOCHS", 128)), val_metric_name="mae",
                             train_metric_name=os.environ.get("FM_REALMLP_LOSS", "pinball(0.5)"))
    Xtr, Xte = impute(Xtr, Xte, cats)
    m.fit(Xtr, y)
    return {"": np.asarray(m.predict(Xte)).reshape(-1)}


def m_tabm(Xtr, y, Xte, cats, seed, threads, **_):
    """TabM-mini with piecewise-linear embeddings, trained on L1 per ensemble member. The median over members is the output."""
    import torch
    import torch.nn.functional as F
    from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins
    from sklearn.preprocessing import QuantileTransformer
    from tabm import TabM

    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    A, B, ci = ordinal(Xtr, Xte, cats)
    nums = [i for i in range(A.shape[1]) if i not in ci]
    rng = np.random.default_rng(seed)
    is_na = np.isnan(A[:, nums]).mean(axis=0) > 0
    fill = np.nanmedian(A[:, nums], axis=0)
    def num(X):
        Z = X[:, nums]
        miss = np.isnan(Z)[:, is_na].astype(np.float32)
        return np.where(np.isnan(Z), fill, Z), miss
    Zt, Mt = num(A)
    Zv, Mv = num(B)
    qt = QuantileTransformer(n_quantiles=min(1000, len(Zt)), output_distribution="normal", subsample=None, random_state=seed).fit(
        Zt + rng.normal(0, 1e-5, Zt.shape) * Zt.std(axis=0, keepdims=True))
    Zt, Zv = np.c_[qt.transform(Zt), Mt].astype(np.float32), np.c_[qt.transform(Zv), Mv].astype(np.float32)
    card = [int(np.nanmax(A[:, i])) + 2 for i in ci]
    def catc(X):
        return torch.as_tensor(np.nan_to_num(X[:, ci], nan=-1).clip(-1, np.array(card) - 3).astype(np.int64) + 1)
    Ct, Cv = catc(A), catc(B)
    mu, sd = np.median(y), np.subtract(*np.percentile(y, [75, 25])) / 1.349
    yt = torch.as_tensor((y - mu) / sd, dtype=torch.float32)
    idx = rng.permutation(len(y))
    n_val = max(200, len(y) // 10)
    va, tr = torch.as_tensor(idx[:n_val]), torch.as_tensor(idx[n_val:])
    part = Zt[idx[n_val:]]
    keep = part.max(axis=0) > part.min(axis=0)   # a near-constant column can be constant in one split, and that breaks the bins
    Zt, Zv = torch.as_tensor(Zt[:, keep]), torch.as_tensor(Zv[:, keep])
    emb = PiecewiseLinearEmbeddings(compute_bins(Zt[tr], n_bins=24), d_embedding=16, activation=False, version="B")
    Zt, Zv, Ct, Cv, yt = (t.to(DEVICE) for t in (Zt, Zv, Ct, Cv, yt))
    net = TabM.make(n_num_features=Zt.shape[1], cat_cardinalities=card, d_out=1, num_embeddings=emb, k=16, n_blocks=2,
                    d_block=256, dropout=0.1, arch_type="tabm-mini").to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=3e-4)

    def run(Z, Cc):
        with torch.no_grad():
            return torch.cat([net(Z[i:i + 2048], Cc[i:i + 2048]).squeeze(-1) for i in range(0, len(Z), 2048)])
    best, best_state, bad, started = np.inf, None, 0, time.time()
    budget = float(os.environ.get("FM_TABM_SECONDS", 420))
    for epoch in range(100):
        net.train()
        for b in torch.as_tensor(rng.permutation(len(tr))).split(256):
            rows = tr[b]
            pred = net(Zt[rows], Ct[rows]).squeeze(-1)
            loss = F.l1_loss(pred, yt[rows][:, None].expand_as(pred))
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        v = (run(Zt[va], Ct[va]).median(dim=1).values - yt[va]).abs().mean().item()
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
        if bad >= 8 or time.time() - started > budget:
            break
    print(f"tabm epochs={epoch + 1} val_l1={best:.4f}", flush=True)
    net.load_state_dict(best_state)
    net.eval()
    return {"": run(Zv, Cv).median(dim=1).values.cpu().numpy() * sd + mu}


MODELS = dict(lgb=m_lgb, cat=m_cat, xgb=m_xgb, knn=m_knn, tabpfn=m_tabpfn, tabicl=m_tabicl, realmlp=m_realmlp, tabm=m_tabm)


# ---- running ---------------------------------------------------------------------------------------------

def needed_rows(test: pd.DataFrame) -> np.ndarray:
    """Mask over the stacked car=1, car=0 rows: the passes that can reach the score.

    A row whose advert parking matches its input parking is scored on its own pass alone. Only flipped or blank
    rows need both. This only saves compute on slow models. LightGBM and CatBoost scored identically with and without it.
    """
    inp = test.input_car_space.astype(float).to_numpy()
    adv = test.advert_car_space.astype(float).fillna(test.input_car_space.astype(float)).to_numpy()
    both = ~((inp > 0) == (adv > 0)) | np.isnan(inp) | np.isnan(adv)
    return np.r_[(adv > 0) | both, ~(adv > 0) | both]


def fill_passes(raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Put predictions for needed rows back in the stack. A skipped pass copies the row's other pass."""
    full = np.full(len(mask), np.nan)
    full[mask] = raw
    h = len(mask) // 2
    w, wo = full[:h], full[h:]
    return np.r_[np.where(np.isnan(w), wo, w), np.where(np.isnan(wo), w, wo)]


def key(model: str, variant: str, form: str, n: int, seed: int, tag: str, val: bool) -> str:
    return "__".join([("val" if val else "B"), model + (f"-{variant}" if variant else ""), form, f"n{n}", f"s{seed}"] + ([tag] if tag else []))


def run(a) -> None:
    os.environ.update(OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads), OPENBLAS_NUM_THREADS=str(a.threads))
    RES.mkdir(parents=True, exist_ok=True)
    panel, cols = load_panel()
    train_all, test = frames(panel, a.val)
    print(f"train {len(train_all)} test {len(test)} cols {len(cols)}", flush=True)
    for seed in [int(s) for s in a.seeds.split(",")]:
        for n in [int(x) for x in str(a.n).split(",")]:
            train = subsample(train_all, n, seed, a.how)
            Xtr, Xte, cats = design(train, test, cols)
            for form in a.targets.split(","):
                y = target(train, form)
                for name in a.models.split(","):
                    done = RES / (key(name, "", form, n, seed, a.tag, a.val) + ".done")
                    if done.exists():
                        continue
                    t0 = time.time()
                    try:
                        mask = needed_rows(test) if a.sparse else np.ones(len(Xte), bool)
                        out = MODELS[name](Xtr, y, Xte[mask].reset_index(drop=True), cats, seed, a.threads, n_estimators=a.n_estimators)
                        out = {k: fill_passes(np.asarray(v, float), mask) for k, v in out.items()}
                    except Exception:  # noqa: BLE001 - one failed model must not stop the queue
                        print(f"FAILED {name} {form} n{n} s{seed}\n{traceback.format_exc()[-1500:]}", flush=True)
                        continue
                    secs = time.time() - t0
                    for variant, raw in out.items():
                        w, wo = to_rent(np.asarray(raw, float), test, form)
                        k = key(name, variant, form, n, seed, a.tag, a.val)
                        np.savez(RES / f"{k}.npz", w=w, wo=wo, secs=secs, n_train=len(train))
                        s = score_arrays(test, w, wo)
                        print(f"{k}: mae={s['mae']:.2f} w100={s['within100']:.3f} {secs:.0f}s", flush=True)
                    done.touch()


def scored_frame(test: pd.DataFrame, w: np.ndarray, wo: np.ndarray) -> pd.DataFrame:
    p = pd.DataFrame({"pred_with": w, "pred_without": wo, "premium": w - wo}, index=test.index)
    prices = M.compose_prices(p, test["input_car_space"])
    return pd.concat([test[["rent", "advert_car_space", "input_car_space"]], prices], axis=1)


def score_arrays(test, w, wo) -> dict:
    return score(scored_frame(test, w, wo))


def abs_err(test, w, wo) -> pd.Series:
    s = scored_prediction(scored_frame(test, w, wo))
    return (s["pred"] - s["rent"]).abs().fillna(np.inf)


def boot_gap(ea: np.ndarray, eb: np.ndarray, reps: int = 2000, seed: int = 11) -> tuple[float, float, float]:
    """Paired bootstrap over test rows of mean-over-seeds median error A minus B. ea, eb are (seeds, rows)."""
    rng = np.random.default_rng(seed)
    n = ea.shape[1]
    gaps = np.empty(reps)
    for r in range(reps):
        i = rng.integers(0, n, n)
        gaps[r] = np.median(ea[:, i], axis=1).mean() - np.median(eb[:, i], axis=1).mean()
    return float(np.median(ea, axis=1).mean() - np.median(eb, axis=1).mean()), *np.percentile(gaps, [2.5, 97.5]).tolist()


def load_groups(val: bool = False) -> dict:
    """{(model, form, n, tag): {seed: (w, wo, secs)}} from the saved predictions."""
    g: dict = {}
    for f in sorted(RES.glob(("val" if val else "B") + "__*.npz")):
        parts = f.stem.split("__")
        name, form, n, s = parts[1], parts[2], parts[3], int(parts[4][1:])
        d = np.load(f)
        g.setdefault((name, form, n, parts[5] if len(parts) > 5 else ""), {})[s] = (d["w"], d["wo"], float(d["secs"]))
    return g


BLENDS = {  # equal weights, fixed before looking at Fold B. Tree pairs are the control: any blend cuts variance.
    "lgb+cat": [("lgb", "log_ratio", "n0"), ("cat", "log_ratio", "n0")],
    "lgb+xgb+cat": [("lgb", "log_ratio", "n0"), ("xgb", "log_ratio", "n0"), ("cat", "log_ratio", "n0")],
    "lgb+tabpfn2k": [("lgb", "log_ratio", "n0"), ("tabpfn-median", "log_ratio", "n2000")],
    "cat+tabpfn2k": [("cat", "log_ratio", "n0"), ("tabpfn-median", "log_ratio", "n2000")],
    "lgb+cat+tabpfn2k": [("lgb", "log_ratio", "n0"), ("cat", "log_ratio", "n0"), ("tabpfn-median", "log_ratio", "n2000")],
    "lgb+tabicl2k": [("lgb", "log_ratio", "n0"), ("tabicl-median", "log_ratio", "n2000")],
    "lgb+tabicl5k": [("lgb", "log_ratio", "n0"), ("tabicl-median", "log_ratio", "n5000")],
    "lgb+tabicl10k": [("lgb", "log_ratio", "n0"), ("tabicl-median", "log_ratio", "n10000")],
    "lgb+cat+tabicl10k": [("lgb", "log_ratio", "n0"), ("cat", "log_ratio", "n0"), ("tabicl-median", "log_ratio", "n10000")],
    "lgb+tabpfn2k+tabicl10k": [("lgb", "log_ratio", "n0"), ("tabpfn-median", "log_ratio", "n2000"), ("tabicl-median", "log_ratio", "n10000")],
    "lgb+cat+tabpfn2k+tabicl10k": [("lgb", "log_ratio", "n0"), ("cat", "log_ratio", "n0"), ("tabpfn-median", "log_ratio", "n2000"),
                                   ("tabicl-median", "log_ratio", "n10000")],
    "cat+tabicl5k": [("cat", "log_ratio", "n0"), ("tabicl-median", "log_ratio", "n5000")],
    "lgb+cat+tabicl5k": [("lgb", "log_ratio", "n0"), ("cat", "log_ratio", "n0"), ("tabicl-median", "log_ratio", "n5000")],
    "lgb+tabicl5k+tabpfn2k": [("lgb", "log_ratio", "n0"), ("tabicl-median", "log_ratio", "n5000"), ("tabpfn-median", "log_ratio", "n2000")],
    "lgb+cat+tabicl5k+tabm": [("lgb", "log_ratio", "n0"), ("cat", "log_ratio", "n0"), ("tabicl-median", "log_ratio", "n5000"), ("tabm", "log_ratio", "n0")],
    "tabicl5k+tabm": [("tabicl-median", "log_ratio", "n5000"), ("tabm", "log_ratio", "n0")],
    "lgb+tabm": [("lgb", "log_ratio", "n0"), ("tabm", "log_ratio", "n0")],
    "tabicl5k+realmlp": [("tabicl-median", "log_ratio", "n5000"), ("realmlp", "log_ratio", "n0")],
    "tabicl5k+realmlp+tabm": [("tabicl-median", "log_ratio", "n5000"), ("realmlp", "log_ratio", "n0"), ("tabm", "log_ratio", "n0")],
    "cat+tabicl5k+realmlp": [("cat", "log_ratio", "n0"), ("tabicl-median", "log_ratio", "n5000"), ("realmlp", "log_ratio", "n0")],
    "lgb+cat+tabicl5k+realmlp+tabm": [("lgb", "log_ratio", "n0"), ("cat", "log_ratio", "n0"), ("tabicl-median", "log_ratio", "n5000"),
                                      ("realmlp", "log_ratio", "n0"), ("tabm", "log_ratio", "n0")],
    "lgb+realmlp": [("lgb", "log_ratio", "n0"), ("realmlp", "log_ratio", "n0")],
}


def blend_preds(groups: dict, members: list) -> dict:
    """{seed: (w, wo, secs)} averaged in dollars over members, for seeds that every member has."""
    parts = [groups.get((*m, "")) for m in members]
    if not all(parts):
        return {}
    seeds = set.intersection(*[set(p) for p in parts])
    return {s: (np.mean([p[s][0] for p in parts], axis=0), np.mean([p[s][1] for p in parts], axis=0), sum(p[s][2] for p in parts))
            for s in sorted(seeds)}


def holdout_weight(panel: pd.DataFrame, a: tuple, b: tuple) -> float | None:
    """Weight on b for a two-model blend, chosen on the pre-cutoff holdout only. Fold B is not read here."""
    g, test = load_groups(val=True), frames(panel, True)[1]
    pa, pb = g.get((*a, "")), g.get((*b, ""))
    if not pa or not pb or not set(pa) & set(pb):
        return None
    seeds, grid = sorted(set(pa) & set(pb)), np.round(np.arange(0, 1.01, 0.1), 1)
    loss = [np.mean([score_arrays(test, (1 - v) * pa[s][0] + v * pb[s][0], (1 - v) * pa[s][1] + v * pb[s][1])["mae"] for s in seeds]) for v in grid]
    print("holdout", a[0], b[0], "seeds", seeds, dict(zip(grid.tolist(), np.round(loss, 2).tolist())))
    return float(grid[int(np.argmin(loss))])


def score_holdout(panel: pd.DataFrame) -> list:
    """The same comparison on the last training origin (adverts of Feb-Mar 2026), trained on the 11 origins before it.

    Fold B has been looked at many times, so this is the less worn check. Nothing here picks a setting.
    """
    test = frames(panel, True)[1]
    g = load_groups(val=True)
    for name, members in BLENDS.items():
        if (b := blend_preds(g, members)):
            g[(name, "blend", "mix", "")] = b
    errs = {k: pd.DataFrame({s: abs_err(test, w, wo) for s, (w, wo, _) in v.items()}) for k, v in g.items()}
    base = {r: errs.get((r, "log_ratio", "n0", "")) for r in ("lgb", "cat")} | {"trees": errs.get(("lgb+cat", "blend", "mix", ""))}
    out = []
    for k, e in errs.items():
        rec = dict(model=k[0], target=k[1], n=k[2], seeds=e.shape[1], mae=float(e.median().mean()), w100=float((e <= 100).mean().mean()))
        for r, b in base.items():
            if b is not None and (k[0], k[2]) != (r, "n0") and k[0] != "lgb+cat":
                c = sorted(set(e.columns) & set(b.columns))
                rec.update(zip((f"gap_{r}", f"{r}_lo", f"{r}_hi"), boot_gap(e[c].to_numpy().T, b[c].to_numpy().T)))
        out.append(rec)
    t = pd.DataFrame(out).sort_values("mae")
    print(f"Holdout window, {len(test)} adverts\n", t.round(2).to_string(index=False))
    return out


def score_all(a) -> None:
    panel, _ = load_panel()
    _, test = frames(panel, False)
    first = test.anchor_own.isna()
    groups = load_groups()
    for name, members in BLENDS.items():
        if (b := blend_preds(groups, members)):
            groups[(name, "blend", "mix", "")] = b
    for x, y in ((("lgb", "log_ratio", "n0"), ("tabpfn-median", "log_ratio", "n2000")), (("lgb", "log_ratio", "n0"), ("cat", "log_ratio", "n0")),
                 (("lgb", "log_ratio", "n0"), ("tabicl-median", "log_ratio", "n2000")), (("lgb", "log_ratio", "n0"), ("tabicl-median", "log_ratio", "n5000")),
                 (("cat", "log_ratio", "n0"), ("tabicl-median", "log_ratio", "n5000")), (("lgb", "log_ratio", "n0"), ("tabicl-median", "log_ratio", "n10000"))):
        v = holdout_weight(panel, x, y)
        if v is not None and (*x, "") in groups and (*y, "") in groups:
            pa, pb = groups[(*x, "")], groups[(*y, "")]
            groups[(f"holdout-w{v}:{x[0]}+{y[0]}-{y[2]}", "blend", "mix", "")] = {
                s: ((1 - v) * pa[s][0] + v * pb[s][0], (1 - v) * pa[s][1] + v * pb[s][1], 0.0) for s in sorted(set(pa) & set(pb))}
    rows, errs = [], {}
    for k, seeds in groups.items():
        e = pd.DataFrame({s: abs_err(test, w, wo) for s, (w, wo, _) in seeds.items()})
        errs[k] = e
        ss = [score_arrays(test, w, wo) for w, wo, _ in seeds.values()]
        f = [score_arrays(test[first], w[first.to_numpy()], wo[first.to_numpy()]) for w, wo, _ in seeds.values()]
        rows.append(dict(model=k[0], target=k[1], n=k[2], tag=k[3], seeds=len(seeds),
                         mae=np.mean([x["mae"] for x in ss]), mae_sd=np.std([x["mae"] for x in ss]),
                         w100=np.mean([x["within100"] for x in ss]), bias=np.mean([x["bias"] for x in ss]),
                         first_mae=np.mean([x["mae"] for x in f]), first_w100=np.mean([x["within100"] for x in f]),
                         secs=np.mean([v[2] for v in seeds.values()])))
    t = pd.DataFrame(rows).sort_values(["n", "mae"])
    pd.set_option("display.width", 250, "display.max_rows", 300, "display.max_columns", 30)
    print(t.round(3).to_string(index=False))

    gaps = []
    for k, e in errs.items():
        ref = ("lgb", "log_ratio", "n0" if k[2] == "mix" else k[2], "")
        if ref not in errs or k == ref:
            continue
        common = sorted(set(e.columns) & set(errs[ref].columns))
        if common:
            g, lo, hi = boot_gap(e[common].to_numpy().T, errs[ref][common].to_numpy().T)
            rec = dict(model=k[0], target=k[1], n=k[2], tag=k[3], gap_vs_lgb=g, lo=lo, hi=hi)
            if k[2] in ("n0", "mix") or k[0].startswith(("tabpfn", "tabicl")):   # first-time flats, against LightGBM on all rows
                f0 = ("lgb", "log_ratio", "n0", "")
                c0 = sorted(set(e.columns) & set(errs[f0].columns))
                mine, base = (x[c0].loc[first[x.index]].to_numpy().T for x in (e, errs[f0]))
                rec.update(zip(("first_gap", "first_lo", "first_hi"), boot_gap(mine, base)))
            full = ("lgb", "log_ratio", "n0", "")   # subsampled models also against LightGBM on all rows, the real competitor
            if k[2] not in ("n0", "mix") and set(e.columns) & set(errs[full].columns):
                c2 = sorted(set(e.columns) & set(errs[full].columns))
                rec.update(zip(("gap_vs_full", "full_lo", "full_hi"), boot_gap(e[c2].to_numpy().T, errs[full][c2].to_numpy().T)))
            cat = ("cat", "log_ratio", "n0", "")   # the strongest single tree reference
            if cat in errs and k != cat:
                c4 = sorted(set(e.columns) & set(errs[cat].columns))
                rec.update(zip(("gap_vs_cat", "cat_lo", "cat_hi"), boot_gap(e[c4].to_numpy().T, errs[cat][c4].to_numpy().T)))
            ctl = ("lgb+cat", "blend", "mix", "")
            if k[1] == "blend" and ctl in errs and k != ctl:
                c3 = sorted(set(e.columns) & set(errs[ctl].columns))
                rec.update(zip(("gap_vs_treeblend", "tree_lo", "tree_hi"), boot_gap(e[c3].to_numpy().T, errs[ctl][c3].to_numpy().T)))
            gaps.append(rec)
    gt = pd.DataFrame(gaps)
    if len(gt):
        print(gt.sort_values(["n", "gap_vs_lgb"]).round(2).to_string(index=False))
    holdout = score_holdout(panel)
    out = RES / "summary.json"
    out.write_text(json.dumps(dict(scores=rows, gaps=gaps, holdout=holdout), indent=1, default=float))
    if a.log:
        keep = {"lgb", "cat", "xgb", "knn-k50", "tabpfn-median", "tabicl-median", "tabm", "realmlp", "lgb+cat", "cat+tabicl5k+realmlp"}
        for r in [x for x in rows if x["model"] in keep and not x["tag"] and x["target"] in ("log_ratio", "blend") and x["n"] in ("n0", "n2000", "n5000", "mix")]:
            log_experiment(f"Foundation lane Fold B: {r['model']} {r['target']} {r['n']} {r['tag']} ({r['seeds']} seeds, mean)".strip(),
                           "FEATURE_COLS:33 cols", f"fm:{r['model']}:{r['target']}:{r['n']}", M.NO_SCORE,
                           dict(mae=r["mae"], within100=r["w100"], r2=float("nan"), bias=float("nan")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--models", required=True)
    r.add_argument("--targets", default="log_ratio,rent")
    r.add_argument("--n", default="0", help="training rows, comma separated; 0 means all")
    r.add_argument("--seeds", default="1,2,3")
    r.add_argument("--threads", type=int, default=2)
    r.add_argument("--n-estimators", type=int, default=2)
    r.add_argument("--how", default="random", choices=["random", "recent"])
    r.add_argument("--tag", default="")
    r.add_argument("--sparse", action="store_true", help="skip test passes that cannot reach the score")
    r.add_argument("--val", action="store_true", help="hold out the last training origin instead of scoring Fold B")
    s = sub.add_parser("score")
    s.add_argument("--log", action="store_true")
    args = ap.parse_args()
    sys.exit(run(args) if args.cmd == "run" else score_all(args))
