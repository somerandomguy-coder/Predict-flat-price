"""Collect every step's rows, audit seed coverage, and bootstrap candidates against a same-seed reference.

usage: methods_table.py [REF:name,name ...]   each group compares its names with REF on Fold B and on all four windows pooled.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from local import methods_lib as L


def rows() -> pd.DataFrame:
    out = []
    for path in sorted(glob.glob(str(L.OUT / "step[3-6]*.json"))):
        if path.endswith(("step3.json", "logbias.json")):  # step3.json is the first base run, kept as step3_base.json
            continue
        out += json.load(open(path))
    return pd.DataFrame(out).drop_duplicates("name")


def audit(t: pd.DataFrame) -> None:
    """A row exists only after all its seeds and windows finished. Check the saved predictions agree."""
    missing = []
    for name in t.name:
        path = L.OUT / f"preds_{name}.npz"
        if path.exists():
            have = {k[:-2] if k.endswith("_w") else k[:-3] for k in np.load(path).files}
            if have != set(L.WINDOWS):
                missing.append((name, sorted(have)))
    print("rows", len(t), "| seeds per row:", t.n_seeds.value_counts(dropna=False).to_dict() if "n_seeds" in t else "n/a")
    print("rows with incomplete windows:", missing or "none")


def compare(panel, ref_name, names) -> None:
    ref = L.load_preds(ref_name)
    print(f"reference {ref_name}")
    for name in names:
        p = L.load_preds(name)
        a_all, b_all = [], []
        for w in L.WINDOWS:
            te = L.split(panel, w)[1]
            a, b = L.abs_err(te, *ref[w]), L.abs_err(te, *p[w])
            if w == "B":
                rb = L.paired_bootstrap(a, b)
                rc = L.paired_bootstrap(a, b, te.building_key, reps=1000)
                wb = 100 * ((b <= 100).mean() - (a <= 100).mean())
            a_all.append(a)
            b_all.append(b)
        a, b = pd.concat(a_all).reset_index(drop=True), pd.concat(b_all).reset_index(drop=True)
        pool = L.paired_bootstrap(a, b)
        print(f"  {name:26s} B {rb['diff']:+.2f} [{rb['lo']:+.2f},{rb['hi']:+.2f}] bldg [{rc['lo']:+.2f},{rc['hi']:+.2f}] w100 {wb:+.2f}pp | "
              f"pooled n={len(a)} {pool['diff']:+.2f} [{pool['lo']:+.2f},{pool['hi']:+.2f}] w100 {100 * ((b <= 100).mean() - (a <= 100).mean()):+.2f}pp", flush=True)


def blend_catboost(panel) -> None:
    """LightGBM and CatBoost averaged from saved predictions. No fitting."""
    lgbm, cb = L.load_preds("blend_log1.00_seeds5"), L.load_preds("s5_cb_d6_r300")
    mixed = {w: tuple((x + y) / 2 for x, y in zip(lgbm[w], cb[w])) for w in L.WINDOWS}
    sc = {w: L.score_pair(L.split(panel, w)[1], *mixed[w]) for w in L.WINDOWS}
    name = "s5_lgbm_cb_half"
    L.save_preds(name, mixed)
    row = dict(name=name, n_seeds=7, **L.summarise(sc))
    row["run_id"] = L.log_row(name, "blend:lgbm_5seed+catboost_2seed", sc["B"], row)
    # First-time flats routed to the direct target, on the five-seed predictions.
    log, direct = lgbm, L.load_preds("blend_log0.00_seeds5")
    routed = {}
    for w in L.WINDOWS:
        first = L.split(panel, w)[1].anchor_own.isna().to_numpy()
        routed[w] = tuple(np.where(first, d, g) for d, g in zip(direct[w], log[w]))
    sc = {w: L.score_pair(L.split(panel, w)[1], *routed[w]) for w in L.WINDOWS}
    L.save_preds("f_route_first_direct", routed)
    row2 = dict(name="f_route_first_direct", n_seeds=5, **L.summarise(sc))
    row2["run_id"] = L.log_row("f_route_first_direct", "lgbm_l1:route:first_time=direct:else=log_ratio:seeds=5", sc["B"], row2)
    (L.OUT / "step5_blend.json").write_text(json.dumps([row, row2], indent=1))


def main():
    panel = L.load_panel()
    if not Path(L.OUT / "step5_blend.json").exists():
        blend_catboost(panel)
    t = rows()
    audit(t)
    cols = [c for c in ("name", "n_seeds", "B_mae", "B_w100", "I_mae", "I_w100", "I1_mae", "I2_mae", "I3_mae", "B_first", "B_repeat", "I_first", "I_repeat") if c in t]
    print(t[cols].sort_values("I_mae").round(3).to_string(index=False))
    for group in sys.argv[1:]:
        ref, names = group.split(":")
        compare(panel, ref, names.split(","))


if __name__ == "__main__":
    main()
