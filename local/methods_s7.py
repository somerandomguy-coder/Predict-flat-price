"""Step 7: the parking gap. Anchors rebuilt per pass, and blends of the two-pass gap with the within-building premium.

Variants of the two passes, all seed-averaged:
  fixed       current model: anchors stay at input parking in both passes
  test_only   train as now, but each test pass gets anchors rebuilt at that pass's parking
  train_test  also rebuild training anchors at the advert's parking, so training rows match a pass.
              This is not the leak fixed in decision 13: both test passes are made without seeing the advert, and
              the scorer picks between them. A training row's own parking is a known fact about a past advert.

Anchors are rebuilt by calling features._prepare and features._add_anchors from here, so src/features.py is unchanged.
Each window's parking inputs stop at that row's own origin.
"""
import json
import sys

import numpy as np
import pandas as pd

from local import methods_lib as L
from src import features as F
from src import parking
from src.index import DEFAULT_PARAMS
from src.load import load_leases
from src.validate import scored_prediction

ANCHOR_OUT = F.ANCHOR_COLS + ["anchor_own_parking", "anchor_own_parking_match", "anchor_bldg_parking_match", "anchor_block_level",
                              "best_anchor", "best_anchor_source"]
_prep = {}


def prep(origin):
    if origin not in _prep:
        leases = load_leases()
        _prep[origin] = F._prepare(leases, origin, DEFAULT_PARAMS)
    return _prep[origin]


def rebuild(df, parking_flag) -> pd.DataFrame:
    """df with its anchor columns rebuilt as if the flat's parking were parking_flag (array, NaN = unknown)."""
    out = df.copy()
    for origin, idx in df.groupby("origin").groups.items():
        X = df.loc[idx].copy()
        X["suburb"] = X.suburb.astype(object)
        X["input_car_space"] = pd.Series(parking_flag, index=df.index).loc[idx]
        X = F._add_anchors(X.reset_index(drop=True), prep(origin), origin)
        X.index = idx
        for c in ANCHOR_OUT:
            out.loc[idx, c] = X[c]
    return out


def check_rebuild(panel) -> None:
    """Rebuilding at the row's own input parking must reproduce the panel exactly."""
    te = L.split(panel, "B")[1]
    again = rebuild(te, te.input_car_space.to_numpy(float))
    for c in ("anchor_own", "anchor_bldg", "anchor_block", "best_anchor"):
        assert np.allclose(again[c].to_numpy(float), te[c].to_numpy(float), equal_nan=True), c
    print("rebuild check passed", flush=True)


def fit_all(tr, cols, params, rounds, seeds):
    cfg = dict(lgb_params={**params, "num_threads": L.THREADS}, num_boost_round=rounds)
    return [L.M.fit_model(tr, cols, "log_ratio", {**cfg, "seed": s}) for s in seeds]


def passes(models, te_with, te_without):
    """Two passes where each test pass may carry its own anchors."""
    return (np.mean([m.predict(te_with, car_space=1) for m in models], axis=0),
            np.mean([m.predict(te_without, car_space=0) for m in models], axis=0))


def flipped_mask(te):
    adv, inp = te.advert_car_space.astype(float), te.input_car_space.astype(float)
    return adv.notna() & inp.notna() & ((adv > 0) != (inp > 0))


def main():
    params = eval(sys.argv[1]) if len(sys.argv) > 1 else {}
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    tag = sys.argv[3] if len(sys.argv) > 3 else "current"
    seeds = L.SEEDS[:3]
    panel = L.load_panel()
    cols = L.columns(panel)
    check_rebuild(panel)

    errs, gaps, overall = {}, {}, {}
    for w in L.WINDOWS:
        tr, te = L.split(panel, w)
        table = parking.estimate_premium(L.WINDOWS[w])
        wb = parking.premium_for(te, table)
        te1, te0 = rebuild(te, np.ones(len(te))), rebuild(te, np.zeros(len(te)))
        tr_adv = rebuild(tr, tr.car_space_resolved.gt(0).astype(float).where(tr.car_space_resolved.notna()).to_numpy(float))
        print(w, "anchors rebuilt, within-building median", round(float(wb.median()), 2), flush=True)
        models, models_adv = fit_all(tr, cols, params, rounds, seeds), fit_all(tr_adv, cols, params, rounds, seeds)
        two = {"fixed": passes(models, te, te), "test_only": passes(models, te1, te0), "train_test": passes(models_adv, te1, te0)}
        for name, (pw, pwo) in two.items():
            model_gap = pd.Series(pw - pwo, index=te.index).clip(lower=0)
            for a in (1.0, 0.75, 0.5, 0.25, 0.0):
                label = f"{name}_gap{a:.2f}"
                prem = a * model_gap + (1 - a) * wb
                errs[w, label] = L.abs_err(te, pw, pwo, prem)
                gaps[w, label] = float(prem.median())
                overall[w, label] = L.score_pair(te, pw, pwo, prem)
            # The midpoint construction of parking.pick_premium, for comparison with decision 15.
            mid = (pw + pwo) / 2
            errs[w, f"{name}_midpoint_wb"] = L.abs_err(te, mid + wb / 2, mid - wb / 2)
            gaps[w, f"{name}_midpoint_wb"] = float(wb.median())
            overall[w, f"{name}_midpoint_wb"] = L.score_pair(te, (mid + wb / 2).to_numpy(), (mid - wb / 2).to_numpy())

    labels = sorted({k[1] for k in errs})
    flips = {w: flipped_mask(L.split(panel, w)[1]) for w in L.WINDOWS}
    groups = {w: L.split(panel, w)[1].building_key for w in L.WINDOWS}

    def flip_err(w, label):
        e = errs[w, label]
        return e[e.index.isin(flips[w][flips[w]].index)]

    ref = "fixed_gap1.00"
    rows = []
    for label in labels:
        b = flip_err("B", label)
        pooled = pd.concat([flip_err(w, label) for w in L.WINDOWS])
        inner = pd.concat([flip_err(w, label) for w in L.INNER])
        ref_b, ref_p = flip_err("B", ref), pd.concat([flip_err(w, ref) for w in L.WINDOWS])
        row = dict(name=label, B_flip_n=len(b), B_flip_mae=float(b.median()), B_flip_w100=float((b <= 100).mean()),
                   B_flip_ci=L.bootstrap_ci(b), B_flip_vs_ref=L.paired_bootstrap(ref_b, b),
                   inner_flip_n=len(inner), inner_flip_mae=float(inner.median()),
                   pooled_flip_n=len(pooled), pooled_flip_mae=float(pooled.median()), pooled_flip_ci=L.bootstrap_ci(pooled),
                   pooled_vs_ref=L.paired_bootstrap(ref_p.reset_index(drop=True), pooled.reset_index(drop=True)),
                   B_gap=gaps["B", label], B_all=overall["B", label]["mae"], B_w100=overall["B", label]["within100"],
                   I_all=float(np.mean([overall[w, label]["mae"] for w in L.INNER])))
        row["run_id"] = L.log_experiment(f"Method lane parking {tag}: {label} (B flipped n={len(b)} MAE {b.median():.2f}; pooled flipped n={len(pooled)} MAE {pooled.median():.2f}; "
                                         f"median gap {row['B_gap']:.2f}; B within100 {row['B_w100']:.4f})", "FEATURE_COLS", f"lgbm_l1:log_ratio:parking:{label}",
                                         L.M.NO_SCORE, overall["B", label])
        rows.append(row)
        print(f"{label:24s} Bflip {row['B_flip_mae']:6.2f} [{row['B_flip_ci']['mae_lo']:.0f},{row['B_flip_ci']['mae_hi']:.0f}] "
              f"d {row['B_flip_vs_ref']['diff']:+6.2f} [{row['B_flip_vs_ref']['lo']:+.1f},{row['B_flip_vs_ref']['hi']:+.1f}] | "
              f"pooled n={row['pooled_flip_n']} {row['pooled_flip_mae']:6.2f} d {row['pooled_vs_ref']['diff']:+6.2f} "
              f"[{row['pooled_vs_ref']['lo']:+.1f},{row['pooled_vs_ref']['hi']:+.1f}] | gap {row['B_gap']:5.1f} B {row['B_all']:.2f} w100 {row['B_w100']:.4f} I {row['I_all']:.2f}", flush=True)
    (L.OUT / f"step7_{tag}.json").write_text(json.dumps(rows, indent=1, default=str))


if __name__ == "__main__":
    main()
