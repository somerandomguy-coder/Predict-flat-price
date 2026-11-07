"""Collect every step's rows into one table, and bootstrap named finalists against the reference on Fold B and the inner windows."""
import glob
import json
import sys

import numpy as np
import pandas as pd

from local import methods_lib as L

REF = "blend_log1.00_seeds5"


def rows() -> pd.DataFrame:
    out = []
    for path in sorted(glob.glob(str(L.OUT / "step[3-6]*.json"))):
        if path.endswith(("step3.json", "logbias.json")):
            continue
        out += json.load(open(path))
    return pd.DataFrame(out).drop_duplicates("name")


def compare(panel, names) -> None:
    ref = L.load_preds(REF)
    for name in names:
        p = L.load_preds(name)
        parts_a, parts_b = [], []
        for w in L.WINDOWS:
            te = L.split(panel, w)[1]
            a, b = L.abs_err(te, *ref[w]), L.abs_err(te, *p[w])
            if w == "B":
                rb = L.paired_bootstrap(a, b)
                rc = L.paired_bootstrap(a, b, te.building_key, reps=1000)
            parts_a.append(a)
            parts_b.append(b)
        a, b = pd.concat(parts_a).reset_index(drop=True), pd.concat(parts_b).reset_index(drop=True)
        pool = L.paired_bootstrap(a, b)
        print(f"{name:28s} B {rb['diff']:+.2f} [{rb['lo']:+.2f},{rb['hi']:+.2f}] bldg [{rc['lo']:+.2f},{rc['hi']:+.2f}] | "
              f"all 4 windows n={len(a)} {pool['diff']:+.2f} [{pool['lo']:+.2f},{pool['hi']:+.2f}] "
              f"w100 {100 * ((b <= 100).mean() - (a <= 100).mean()):+.2f}pp", flush=True)


def main():
    t = rows()
    cols = [c for c in ("name", "B_mae", "B_w100", "I_mae", "I_w100", "I1_mae", "I2_mae", "I3_mae", "B_seed_sd", "B_first", "B_repeat", "I_first", "I_repeat") if c in t]
    print(t[cols].sort_values("I_mae").round(4).to_string(index=False))
    if len(sys.argv) > 1:
        compare(L.load_panel(), sys.argv[1:])


if __name__ == "__main__":
    main()
