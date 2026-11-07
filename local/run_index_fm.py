"""Time-series foundation models for the monthly suburb index, against holding the index flat.

For each rolling origin up to the Fold B cutoff the index is fitted on leases dated by that origin, and a model
sees only those monthly values. Only monthly aggregates reach a model, and inference is local. Leases after the
end of Fold B are dropped straight after loading, so Fold A months are never used, even as truth.

Three checks:
  index    forecast of the log index change, against the index refitted two months later
  anchor   best_anchor x forecast gain, scored on the adverts at each origin
  feature  the forecast gain as one extra LightGBM feature, on Fold B, 3 seeds
"""
import json
import os
import time

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from local.run_foundation import CUTOFF, RES, boot_gap, design, frames, load_panel, score_arrays, abs_err, target, to_rent
from src import config as C
from src import models as M
from src.index import rent_index
from src.load import load_leases

H = 2
FOLD_END = C.FOLDS["B"]["end"]


def history(curves: np.ndarray) -> list[np.ndarray]:
    return [100 * np.exp(c) for c in curves]


def forecast(pipe, series: list[np.ndarray]) -> np.ndarray:
    """(series, H) log change from the last observed month, from the model's median."""
    if type(pipe).__name__ == "Chronos2Pipeline":
        q, _ = pipe.predict_quantiles([torch.tensor(s[None, :], dtype=torch.float32) for s in series],
                                      prediction_length=H, quantile_levels=[0.5])
        med = np.stack([np.asarray(x).reshape(H) for x in q])
    else:
        q, _ = pipe.predict_quantiles([torch.tensor(s, dtype=torch.float32) for s in series],
                                      prediction_length=H, quantile_levels=[0.5])
        med = np.asarray(q)[..., 0]
    return np.log(med / np.array([s[-1] for s in series])[:, None])


def baselines(curves: np.ndarray) -> dict:
    last = curves[:, -1:]
    return {"flat": np.zeros((len(curves), H)),
            "seasonal_naive": curves[:, -12:-12 + H] - curves[:, -13:-12],   # same months last year
            "drift_6m": np.c_[[(last[:, 0] - curves[:, -7]) / 6 * h for h in range(1, H + 1)]].T}


def main() -> None:
    torch.set_num_threads(int(os.environ.get("FM_THREADS", 1)))
    from chronos import BaseChronosPipeline
    leases = load_leases()
    leases = leases[leases.lease_date <= FOLD_END]
    panel, cols = load_panel()
    origins = sorted(panel.origin.unique())
    print("origins", len(origins), origins[0], origins[-1], flush=True)
    pipes = {"chronos2": BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cpu"),
             "chronos_bolt_small": BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map="cpu")}

    gains, rows = {}, []
    for o in origins:
        t0 = time.time()
        o = pd.Timestamp(o)
        idx = rent_index(leases, o)
        f = baselines(idx.curves)
        for name, pipe in pipes.items():
            f[name] = forecast(pipe, history(idx.curves))
        later = rent_index(leases, o + pd.offsets.MonthEnd(H))
        true = np.full((len(idx.curves), H), np.nan)
        ends = [o + pd.offsets.MonthEnd(h) for h in range(1, H + 1)]
        for j, e in enumerate(ends):
            mid = pd.DatetimeIndex([e - pd.Timedelta(days=14)])
            ref = pd.DatetimeIndex([o - pd.Timedelta(days=14)])
            for g, sub in enumerate(list(idx.groups) + [None]):
                s = None if sub is None else [sub]
                true[g, j] = later.at(mid, s)[0] - later.at(ref, s)[0]
        for name, v in f.items():
            gains[(o, name)] = (idx, v)
            rows.append(dict(origin=o, model=name, abs_err_h1=np.nanmedian(np.abs(v[:, 0] - true[:, 0])),
                             abs_err_h2=np.nanmedian(np.abs(v[:, 1] - true[:, 1])),
                             all_suburb_err_h2=abs(v[-1, 1] - true[-1, 1]), forecast_all_h2=v[-1, 1], true_all_h2=true[-1, 1]))
        print(o.date(), f"T={idx.curves.shape[1]} G={len(idx.groups)} {time.time() - t0:.0f}s", flush=True)

    t = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    summary = t.groupby("model")[["abs_err_h1", "abs_err_h2", "all_suburb_err_h2"]].mean()
    print("Index error in log points, mean over origins\n", (summary * 100).round(3).sort_values("abs_err_h2"))
    print("Fold B origin\n", (t[t.origin == CUTOFF].set_index("model")[["abs_err_h2", "forecast_all_h2", "true_all_h2"]] * 100).round(3))

    def gain_for(df: pd.DataFrame, name: str) -> np.ndarray:
        out = np.zeros(len(df))
        h = ((df.target_date.dt.year - df.origin.dt.year) * 12 + df.target_date.dt.month - df.origin.dt.month).clip(1, H).to_numpy() - 1
        for o, part in df.groupby("origin"):
            idx, v = gains[(pd.Timestamp(o), name)]
            m = (df.origin == o).to_numpy()
            out[m] = v[idx.codes(part.suburb), h[m]]
        return out

    # anchor check: pooled over every origin, and Fold B on its own
    anchor, test_rows = [], M.fold_rows(panel, "B")
    pool = panel[panel.best_anchor > 0]
    for name in t.model.unique():
        rec = dict(model=name)
        for label, d in (("all_origins", pool), ("foldB", test_rows)):
            p = d.best_anchor.to_numpy(float) * np.exp(gain_for(d, name))
            s = score_arrays(d, p, p)
            rec[f"{label}_mae"], rec[f"{label}_w100"], rec[f"{label}_bias"] = s["mae"], s["within100"], s["bias"]
        anchor.append(rec)
    a = pd.DataFrame(anchor).sort_values("all_origins_mae")
    print("Anchor x forecast gain\n", a.round(3).to_string(index=False))

    # feature check: does the forecast help LightGBM, as a feature or as a post-hoc shift?
    train, test = frames(panel, False)
    out, errs = [], {}
    for name in ["flat", "chronos2", "chronos_bolt_small", "seasonal_naive"]:
        for seed in (1, 2, 3):
            Xtr, Xte, cats = design(train, test, cols)
            if name != "flat":
                Xtr = Xtr.assign(fm_gain=gain_for(train, name))
                Xte = Xte.assign(fm_gain=np.r_[gain_for(test, name), gain_for(test, name)])
            b = lgb.train({**M.LGB_PARAMS, "seed": seed, "num_threads": 1}, lgb.Dataset(Xtr, target(train, "log_ratio"), categorical_feature=cats),
                          num_boost_round=M.NUM_BOOST_ROUND)
            w, wo = to_rent(b.predict(Xte), test, "log_ratio")
            s = score_arrays(test, w, wo)
            errs.setdefault(name, []).append(abs_err(test, w, wo).to_numpy())
            out.append(dict(model=name, seed=seed, mae=s["mae"], w100=s["within100"], bias=s["bias"]))
    f = pd.DataFrame(out).groupby("model")[["mae", "w100", "bias"]].mean()
    print("LightGBM + forecast gain feature, Fold B\n", f.round(3))
    gaps = {k: boot_gap(np.array(v), np.array(errs["flat"])) for k, v in errs.items() if k != "flat"}
    print("gap vs flat (gap, lo, hi)", {k: tuple(round(x, 2) for x in v) for k, v in gaps.items()})
    (RES / "index_fm.json").write_text(json.dumps(dict(index=(summary * 100).reset_index().to_dict("records"), anchor=anchor,
                                                       feature=f.reset_index().to_dict("records"), gaps=gaps), indent=1, default=float))


if __name__ == "__main__":
    main()
