"""Confirms the picked feature set on fresh seeds, with a paired bootstrap over Fold B adverts. Fold A is never scored.

The set was picked as the best of 13 on 3 seeds, so its first result is biased in its favour. New seeds remove seed
luck but not the luck of these particular adverts; the bootstrap shows how large that is.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config as C  # noqa: E402
from src import external as E  # noqa: E402
from src import models as M  # noqa: E402
from src.features import FEATURE_COLS  # noqa: E402
from src.validate import scored_prediction  # noqa: E402

SEEDS = range(int(sys.argv[3]), int(sys.argv[3]) + 10)
RESAMPLES = 1000
G = E.GROUPS
SETS = {"baseline": [], "strata+osm": G["gnaf_strata"] + G["osm"]}

panel = E.attach(pd.read_parquet(sys.argv[1]), vintage=sys.argv[2])
test = M.fold_rows(panel, "B")
train = M.train_rows(panel, C.FOLDS["B"]["cutoff"], M.N_ORIGINS)
errs, heads = {}, {}
with tempfile.TemporaryDirectory() as tmp:
    for name, extra in SETS.items():
        e, h = [], []
        for seed in SEEDS:
            cfg = dict(name="pd_confirm", frame=panel, feature_cols=FEATURE_COLS + extra, methods=["log_ratio"], seed=seed, log=False,
                       lgb_params=dict(num_threads=2), out_path=Path(tmp) / "r.json")
            res = M.run_experiment(cfg)
            assert "A" not in res["folds"]
            h.append(res["folds"]["B"]["mae"])
            model = M.fit_model(train, res["features"], "log_ratio", cfg)
            prices = M.compose_prices(M.predict_two_pass(model, test), test["input_car_space"])
            s = scored_prediction(pd.concat([test[["rent", "advert_car_space", "input_car_space", "anchor_own"]], prices], axis=1))
            e.append((s.pred - s.rent).abs().to_numpy())
        errs[name], heads[name] = np.array(e), np.array(h)
        print(name, "seeds", len(h), "mean", round(float(np.mean(h)), 2), "sd", round(float(np.std(h, ddof=1)), 2), flush=True)

first = s.anchor_own.isna().to_numpy()
rng = np.random.default_rng(7)
for label, mask in (("all", np.ones(len(first), bool)), ("first-time", first), ("repeat", ~first)):
    a, b = errs["baseline"][:, mask], errs["strata+osm"][:, mask]
    point = np.median(b, axis=1).mean() - np.median(a, axis=1).mean()
    n = mask.sum()
    d = np.empty(RESAMPLES)
    for i in range(RESAMPLES):
        k = rng.integers(0, n, n)
        d[i] = np.median(b[:, k], axis=1).mean() - np.median(a[:, k], axis=1).mean()
    print(f"{label:11s} n={n} baseline {np.median(a, axis=1).mean():.2f} strata+osm {np.median(b, axis=1).mean():.2f} "
          f"gap {point:+.2f} 90% interval [{np.quantile(d, .05):+.2f}, {np.quantile(d, .95):+.2f}] share of resamples better {np.mean(d < 0):.2f}", flush=True)
# Routed: the richer model only for flats with no earlier advert. Chosen after seeing Fold B, so it is not a clean test.
a, b = errs["baseline"], errs["strata+osm"]
routed = np.where(first[None, :], b, a)
point = np.median(routed, axis=1).mean() - np.median(a, axis=1).mean()
d = np.empty(RESAMPLES)
for i in range(RESAMPLES):
    k = rng.integers(0, a.shape[1], a.shape[1])
    d[i] = np.median(routed[:, k], axis=1).mean() - np.median(a[:, k], axis=1).mean()
print(f"routed      n={a.shape[1]} baseline {np.median(a, axis=1).mean():.2f} routed {np.median(routed, axis=1).mean():.2f} gap {point:+.2f} "
      f"90% interval [{np.quantile(d, .05):+.2f}, {np.quantile(d, .95):+.2f}] share of resamples better {np.mean(d < 0):.2f} "
      f"within100 {(a <= 100).mean():.3f} -> {(routed <= 100).mean():.3f}", flush=True)
print("RUN COMPLETE", flush=True)
