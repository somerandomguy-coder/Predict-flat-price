"""Fit one zoo member on one window and save both parking passes. No scoring here.
usage: python local/zoo/member.py <member> <base|raw> <window>[,<window>...]"""
import os, sys, time, json
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from local.zoo import lib as Z
from src import models as M

CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
SEEDS = (7, 11, 23)
Y = lambda tr: np.log(tr.rent.to_numpy(float) / tr.best_anchor.to_numpy(float))


def cats_of(tr, cols):
    return {c: sorted(tr[c].dropna().astype(str).unique()) for c in cols if c in M.CATEGORICAL}


def lgb_fit(tr, cols, params, seed, y=None, rounds=400):
    import lightgbm as lgb
    cats = cats_of(tr, cols)
    p = {**M.LGB_PARAMS, "num_threads": CPUS, "seed": seed, **params}
    d = lgb.Dataset(M._matrix(tr, cols, cats), Y(tr) if y is None else y, categorical_feature=list(cats) or "auto")
    b = lgb.train(p, d, num_boost_round=rounds)
    return lambda te, car: b.predict(M._matrix(te, cols, cats).assign(car_space_resolved=float(car)))


def ratio(fit_many):
    """Average seeds in log(rent / anchor), then return dollars."""
    def run(tr, te, cols):
        out = {}
        fns = fit_many(tr, cols)
        for car in (1, 0):
            out[car] = te.best_anchor.to_numpy(float) * np.exp(np.mean([f(te, car) for f in fns], axis=0))
        return out
    return run


def lgb_member(params, rounds=400):
    return ratio(lambda tr, cols: [lgb_fit(tr, cols, params, s, rounds=rounds) for s in SEEDS])


def lgb_direct(tr, te, cols):
    fns = [lgb_fit(tr, cols, {}, s, y=tr.rent.to_numpy(float)) for s in SEEDS]
    return {car: np.mean([f(te, car) for f in fns], axis=0) for car in (1, 0)}


def lgb_split(tr, te, cols):
    """Separate models for flats with their own past rent and first-time flats."""
    out = {1: np.zeros(len(te)), 0: np.zeros(len(te))}
    for has in (True, False):
        t, e = tr[tr.anchor_own.notna() == has], te[te.anchor_own.notna() == has]
        c = cols if has else [x for x in cols if x not in Z.RAW_COLS and not x.startswith("anchor_own")]
        p = dict(num_leaves=31, min_data_in_leaf=30)
        fns = [lgb_fit(t, c, p, s) for s in SEEDS]
        m = (te.anchor_own.notna() == has).to_numpy()
        for car in (1, 0):
            out[car][m] = e.best_anchor.to_numpy(float) * np.exp(np.mean([f(e, car) for f in fns], axis=0))
    return out


def cat_member(extra=()):
    def run(tr, te, cols):
        from catboost import CatBoostRegressor, Pool
        names = list(cols) + list(extra)
        catc = ["suburb"] + list(extra)
        def frame(d, car=None):
            x = pd.DataFrame({c: d[c].astype(object).where(d[c].notna(), "NA").astype(str) if c in catc else pd.to_numeric(d[c], errors="coerce").astype(float) for c in names})
            if car is not None:
                x["car_space_resolved"] = float(car)
            return x
        ps = {1: [], 0: []}
        for s in SEEDS[:2]:
            m = CatBoostRegressor(loss_function="MAE", iterations=1000, learning_rate=.06, depth=8, random_seed=s, thread_count=CPUS, verbose=0, allow_writing_files=False)
            m.fit(Pool(frame(tr), Y(tr), cat_features=catc))
            for car in (1, 0):
                ps[car].append(m.predict(Pool(frame(te, car), cat_features=catc)))
        return {c: te.best_anchor.to_numpy(float) * np.exp(np.mean(ps[c], axis=0)) for c in ps}
    return run


def xgb_member(tr, te, cols):
    import xgboost as xgb
    cats = cats_of(tr, cols)
    ps = {1: [], 0: []}
    for s in SEEDS:
        m = xgb.XGBRegressor(objective="reg:pseudohubererror", huber_slope=.1, n_estimators=700, learning_rate=.04, max_depth=7, subsample=.8, colsample_bytree=.8,
                             min_child_weight=1, reg_lambda=1, tree_method="hist", enable_categorical=True, base_score=0.0, random_state=s, n_jobs=CPUS)
        m.fit(M._matrix(tr, cols, cats), Y(tr))
        for car in (1, 0):
            ps[car].append(m.predict(M._matrix(te, cols, cats).assign(car_space_resolved=float(car))))
    return {c: te.best_anchor.to_numpy(float) * np.exp(np.mean(ps[c], axis=0)) for c in ps}


def et_member(tr, te, cols):
    from sklearn.ensemble import ExtraTreesRegressor
    D = Z.Dense(tr, cols, scale=False)
    m = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=3, max_features=.5, n_jobs=CPUS, random_state=7).fit(D(tr), np.clip(Y(tr), -.6, .6))
    return {c: te.best_anchor.to_numpy(float) * np.exp(m.predict(D(te, c))) for c in (1, 0)}


def qreg_member(tr, te, cols):
    import statsmodels.api as sm
    D = Z.Dense(tr, cols)
    X = D(tr).to_numpy(float)
    res = sm.QuantReg(Y(tr), sm.add_constant(X, has_constant="add")).fit(q=.5, max_iter=400)
    return {c: te.best_anchor.to_numpy(float) * np.exp(res.predict(sm.add_constant(D(te, c).to_numpy(float), has_constant="add"))) for c in (1, 0)}


def knn_member(tr, te, cols):
    """Comparables: median log ratio of the 40 nearest adverts in a weighted location, size and anchor space."""
    from sklearn.neighbors import NearestNeighbors
    use = [c for c in ("latitude", "longitude", "bedrooms", "bathrooms", "car_space_resolved", "best_anchor", "anchor_bldg", "anchor_own", "anchor_own_age_days", "floor_level", "raw_prev_ratio") if c in cols]
    w = dict(latitude=2, longitude=2, bedrooms=3, car_space_resolved=2, best_anchor=3)
    D = Z.Dense(tr, cols, onehot=False)
    pick = [c for c in D(tr.head(2)).columns if c in use]
    wv = np.array([w.get(c, 1) for c in pick], np.float32)
    nn = NearestNeighbors(n_neighbors=40, n_jobs=CPUS).fit(D(tr)[pick].to_numpy() * wv)
    y = np.clip(Y(tr), -.6, .6)
    return {c: te.best_anchor.to_numpy(float) * np.exp(np.median(y[nn.kneighbors(D(te, c)[pick].to_numpy() * wv, return_distance=False)], axis=1)) for c in (1, 0)}


QS = np.array([.1, .25, .4, .5, .6, .75, .9])


def qdecode(tr, te, cols, half=40.0):
    """Quantile heads, then the $10-grid point that holds the most predicted mass within +-half dollars."""
    fns = [lgb_fit(tr, cols, dict(objective="quantile", alpha=float(q), num_leaves=31, min_data_in_leaf=30), 7, rounds=300) for q in QS]
    anchor = te.best_anchor.to_numpy(float)
    out = {}
    for car in (1, 0):
        Q = np.sort(anchor[:, None] * np.exp(np.column_stack([f(te, car) for f in fns])), axis=1)
        P = np.tile(QS, (len(Q), 1))
        lo = Q[:, [0]] - (Q[:, [1]] - Q[:, [0]]); hi = Q[:, [-1]] + (Q[:, [-1]] - Q[:, [-2]])
        Qx, Px = np.hstack([lo, Q, hi]), np.hstack([np.zeros((len(Q), 1)), P, np.ones((len(Q), 1))])
        best, bm = Q[:, 3].copy(), np.full(len(Q), -1.0)
        base = np.round(Q[:, 3] / 10) * 10
        cdf = lambda x: np.array([np.interp(x[i], Qx[i], Px[i]) for i in range(len(x))])
        for k in range(-8, 9):
            c = base + 10 * k
            mass = cdf(c + half) - cdf(c - half) - abs(k) * 1e-4
            take = mass > bm
            best[take], bm[take] = c[take], mass[take]
        out[car] = best
    return out


def pytabkit_member(kind):
    def run(tr, te, cols):
        import torch, tempfile
        from pytabkit import RealMLP_TD_Regressor, TabM_D_Regressor
        D = Z.Dense(tr, cols)
        cls = dict(realmlp=RealMLP_TD_Regressor, tabm=TabM_D_Regressor)[kind]
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        m = cls(device=dev, random_state=7, n_cv=1, n_refit=0, val_metric_name="mae", verbosity=0, tmp_folder=tempfile.mkdtemp(dir=os.environ.get("TMPDIR", "/tmp")))
        m.fit(D(tr), np.clip(Y(tr), -.6, .6))
        return {c: te.best_anchor.to_numpy(float) * np.exp(np.asarray(m.predict(D(te, c))).ravel()) for c in (1, 0)}
    return run


def tabpfn_member(tr, te, cols):
    """Local TabPFN only; rows never leave this machine. Context is the most recent 10,000 training adverts."""
    import torch
    from tabpfn import TabPFNRegressor
    D = Z.Dense(tr, cols, onehot=False)
    t = tr.sort_values("target_date").tail(10000)
    Xd = D(t); ci = [Xd.columns.get_loc("suburb_code")]
    m = TabPFNRegressor(device="cuda" if torch.cuda.is_available() else "cpu", n_estimators=4, categorical_features_indices=ci, ignore_pretraining_limits=True, random_state=7)
    m.fit(Xd.to_numpy(), np.clip(Y(t), -.6, .6))
    def pred(c):
        X = D(te, c).to_numpy()
        return np.concatenate([np.asarray(m.predict(X[i:i + 4000], output_type="median")).ravel() for i in range(0, len(X), 4000)])
    return {c: te.best_anchor.to_numpy(float) * np.exp(pred(c)) for c in (1, 0)}


MEMBERS = dict(lgb_l1=lgb_member({}), lgb_huber=lgb_member(dict(objective="huber", alpha=.1)), lgb_dir=lgb_direct, lgb_split=lgb_split,
               cat_mae=cat_member(), cat_bldg=cat_member(("building_key", "mb")), xgb_ph=xgb_member, et=et_member, qreg=qreg_member, knn=knn_member,
               qdecode=qdecode, lgb_n6=lgb_member({}), lgb_n18=lgb_member({}), cat_n6=cat_member(("building_key", "mb")), cat_n18=cat_member(("building_key", "mb")), realmlp=pytabkit_member("realmlp"), tabm=pytabkit_member("tabm"), tabpfn=tabpfn_member)


def main():
    name, fs, windows = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
    panel, final = Z.load()
    cols = Z.columns(panel, fs)
    d = Z.PRED / f"{name}__{fs}"; d.mkdir(parents=True, exist_ok=True)
    for w in windows:
        out = d / f"{w}.parquet"
        if out.exists():
            continue
        t0 = time.time()
        tr, te = Z.split(panel, final, w, int(name.rsplit("_n", 1)[1]) if "_n" in name and name.rsplit("_n", 1)[1].isdigit() else Z.N_ORIGINS)
        p = MEMBERS[name](tr, te, cols)
        r = pd.DataFrame({"gnaf_id": te.gnaf_id.astype(str).to_numpy(), "target_date": te.target_date.to_numpy(), "pred_with": p[1], "pred_without": p[0]})
        assert np.isfinite(r.pred_with).all() and np.isfinite(r.pred_without).all(), "non-finite prediction"
        tmp = out.with_suffix(".tmp"); r.to_parquet(tmp); tmp.replace(out)
        print(name, fs, w, "train", len(tr), "test", len(te), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
