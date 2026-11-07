"""Residual EDA of the current model over six windows that end before Fold B. Fold A and B rows are dropped on read.
Prints aggregates only; row-level out-of-sample predictions stay in out/real/eda/. Run from the repo root with PYTHONPATH=."""
import time
from pathlib import Path
import numpy as np, pandas as pd
from src import config as C, features as F, models as M
from src.predict import build_panel
from src.load import load_leases
from src.validate import scored_prediction

pd.set_option("display.width", 180, "display.max_rows", 200, "display.max_columns", 40)
def h(t): print(f"\n## {t}", flush=True)
PANEL = C.OUT_DIR / "panel_n24.parquet"
WINDOWS = [pd.Timestamp(x) for x in ("2026-01-31", "2025-11-30", "2025-09-30", "2025-07-31", "2025-05-31", "2025-03-31")]
SEEDS = (7, 11, 23)
t0 = time.time()
if PANEL.exists():
    panel = pd.read_parquet(PANEL)
else:
    panel = build_panel(24); panel.to_parquet(PANEL)
print(f"panel rows {len(panel):,} in {time.time()-t0:.0f}s, origins {panel.origin.nunique()}", flush=True)
panel = panel.assign(origin=pd.to_datetime(panel.origin), target_date=pd.to_datetime(panel.target_date))
panel.attrs["feature_cols"] = F.FEATURE_COLS
panel = panel[panel.target_date <= "2026-03-31"]
cols, _ = M.resolve_features(panel, dict(feature_cols=F.FEATURE_COLS))

parts, gains = [], []
for cut in WINDOWS:
    te, tr = panel[panel.origin == cut], M.train_rows(panel, cut, 12)
    ps = []
    for s in SEEDS:
        m = M.fit_model(tr, cols, "log_ratio", dict(seed=s, lgb_params=dict(num_threads=8)))
        ps.append(M.predict_two_pass(m, te))
    gains.append(pd.Series(m.booster.feature_importance("gain"), index=m.cols))
    p = sum(ps) / len(ps); p["premium"] = p.pred_with - p.pred_without
    prices = M.compose_prices(p, te["input_car_space"])
    parts.append(pd.concat([te, prices], axis=1).assign(window=cut, n_train=len(tr)))
    print(cut.date(), "train", len(tr), "test", len(te), f"{time.time()-t0:.0f}s", flush=True)
D = scored_prediction(pd.concat(parts, ignore_index=True))
D["err"] = D.pred - D.rent; D["ae"] = D.err.abs(); D["ape"] = D.ae / D.rent
D.to_parquet(C.OUT_DIR / "eda" / "oos_lgbm.parquet")
med = D.ae.median()

h("1. Per-window score, current model, 3-seed mean")
print(D.groupby("window").agg(n=("ae", "size"), med_ae=("ae", "median"), w100=("ae", lambda s: (s <= 100).mean()), bias=("err", "median"), mean_ae=("ae", "mean")).round(3).to_string())
rng = np.random.default_rng(0)
bs = [D.ae.sample(len(D), replace=True, random_state=int(rng.integers(1e9))).median() for _ in range(300)]
b1700 = [D.ae.sample(1699, replace=True, random_state=int(rng.integers(1e9))).median() for _ in range(300)]
print(f"pooled MedAE {med:.2f}, SE {np.std(bs):.2f}; SE at 1,699 adverts {np.std(b1700):.2f}; median pct err {D.ape.median():.2%}")

h("2. Error by segment, with leverage: pooled median if this segment's errors fell 15%")
D["hist"] = np.where(D.anchor_own.notna(), "has own rent", "first-time")
D["src"] = D.best_anchor_source.map({0: "own", 1: "bldg", 2: "block"})
D["own_age"] = pd.cut(D.anchor_own_age_days / 365.25, [0, .5, 1, 2, 3, 5, 30], labels=["<6m", "6-12m", "1-2y", "2-3y", "3-5y", ">5y"]).astype(object).fillna("none")
D["beds"] = D.bedrooms.clip(upper=4); D["band"] = pd.cut(D.rent, [0, 500, 650, 800, 1000, 1300, 1800, 99999]).astype(str)
D["pband"] = pd.cut(D.pred, [0, 500, 650, 800, 1000, 1300, 1800, 99999]).astype(str)
D["flip"] = np.where(D.advert_car_space.notna() & D.input_car_space.notna() & (D.advert_car_space != D.input_car_space), "flip", "same")
D["bsize"] = pd.cut(D.flats_in_building, [0, 10, 50, 150, 400, 5000]).astype(str)
D["bldg_n"] = pd.cut(D.anchor_bldg_n, [-1, 0, 3, 10, 30, 1e6]).astype(str)
wl = load_leases()[["gnaf_id", "lease_date", "was_leased"]].drop_duplicates(["gnaf_id", "lease_date"])
D = D.merge(wl, left_on=["gnaf_id", "target_date"], right_on=["gnaf_id", "lease_date"], how="left")  # a scoring slice only, never a feature
def seg(col):
    out = []
    for k, g in D.groupby(col, observed=True):
        ae = D.ae.where(D[col] != k, D.ae * .85)
        out.append(dict(seg=k, n=len(g), share=round(len(g) / len(D), 3), med_ae=round(g.ae.median(), 1), med_pct=round(g.ape.median() * 100, 1),
                        w100=round((g.ae <= 100).mean(), 3), bias=round(g.err.median(), 1), overall_if_15pct_better=round(ae.median(), 2)))
    print(f"\n[{col}]\n" + pd.DataFrame(out).to_string(index=False))
for c in ("was_leased", "hist", "src", "own_age", "beds", "pband", "band", "flip", "bsize", "bldg_n"):
    seg(c)
s = D.groupby("suburb", observed=True).agg(n=("ae", "size"), med_ae=("ae", "median"), pct=("ape", "median"), bias=("err", "median")).query("n>=80").sort_values("pct").round(3)
print("\nsuburbs, best and worst by pct error\n", pd.concat([s.head(5), s.tail(6)]).to_string())

h("3. Is residual structure left? r = log(rent/pred)")
D["r"] = np.log(D.rent / D.pred)
print("r quantiles", D.r.quantile([.01, .05, .25, .5, .75, .95, .99]).round(3).to_dict(), "skew", round(D.r.skew(), 2), "kurt", round(D.r.kurt(), 1))
print("MAD-based sd", round(1.4826 * (D.r - D.r.median()).abs().median(), 3), "actual sd", round(D.r.std(), 3), "-> heavy tails if sd >> MAD sd")
order = sorted(D.window.unique())
rows = []
for key in ("building_key", "mb", "suburb"):
    for lag_name, past in (("same window, others", False), ("previous windows only", True)):
        xs, ys = [], []
        for i, w in enumerate(order):
            cur = D[D.window == w]
            if past:
                if i == 0: continue
                prev = D[D.window < w]
                est = cur[key].astype(object).map(prev.groupby(key, observed=True).r.median())
                n = cur[key].astype(object).map(prev.groupby(key, observed=True).r.size())
            else:
                g = cur.groupby(key, observed=True).r
                n = g.transform("size") - 1; est = (g.transform("sum") - cur.r) / n.where(n > 0)
            ok = est.notna() & (n >= 3)
            xs.append(est[ok]); ys.append(cur.r[ok])
        x, y = pd.concat(xs).astype(float), pd.concat(ys).astype(float)
        c = np.clip(x, -.3, .3); yc = np.clip(y, -.3, .3)
        rows.append(dict(key=key, source=lag_name, cover=round(len(x) / len(D), 3), spearman=round(x.corr(y, method="spearman"), 3),
                         slope=round(np.polyfit(c, yc, 1)[0], 3)))
print(pd.DataFrame(rows).to_string(index=False))

h("4. Past-window residual correction, the cheapest possible stacking test (previous windows only)")
out = []
for shrink in (0, 10, 30):
    tot = []
    for i, w in enumerate(order):
        if i == 0: continue
        cur, prev = D[D.window == w], D[D.window < w]
        g = prev.groupby("building_key", observed=True).r.agg(["sum", "size"])
        adj = cur.building_key.astype(object).map(g["sum"] / (g["size"] + shrink)).fillna(0).astype(float) if shrink else 0
        tot.append((cur.pred * np.exp(adj) - cur.rent).abs())
    out.append((shrink, round(pd.concat(tot).median(), 2)))
print("building shrink prior -> MedAE on later windows (0 = no correction):", out)

h("5. Rounding and exact hits")
print("advert on $10 grid", round((D.rent % 10 == 0).mean(), 3), "$50 grid", round((D.rent % 50 == 0).mean(), 3))
for g in (5, 10):
    print(f"round pred to ${g}: MedAE {(np.round(D.pred / g) * g - D.rent).abs().median():.2f}")

h("6. Train rows vs. the final-rows mix: blank rates in OOS rows")
print(D[["anchor_own", "anchor_bldg", "floor_level", "block_homes_2026", "bathrooms"]].isna().mean().round(3).to_dict())

h("7. LightGBM gain share, last window")
g = gains[-1]; print((g / g.sum()).sort_values(ascending=False).head(14).round(3).to_dict())
print(f"done {time.time()-t0:.0f}s")
