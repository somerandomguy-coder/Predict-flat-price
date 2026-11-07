"""Steps 1 and 2: the noise floor, seed averaging, and blending the two target forms."""
import itertools
import json
import time

import numpy as np
import pandas as pd

from local import methods_lib as L
from src import features as F
from src import models as M

N_SEEDS = 10
ALL_SEEDS = L.SEEDS + (202, 303, 404, 505, 606)


def main():
    panel = L.load_panel()
    cols = L.columns(panel)
    print(len(cols), "features", flush=True)

    # The current model through the real entry point, to tie this harness to the published $42.62.
    t = time.time()
    r = M.run_experiment(dict(name="methods_repro", frame=panel, feature_cols=F.FEATURE_COLS, lgb_params=dict(num_threads=L.THREADS),
                              description="Method lane: repro of current model, seed 7, 3 threads"))
    print("repro", r["method_comparison"]["B"]["log_ratio"], r["method_comparison"]["B"]["direct"], round(time.time() - t), flush=True)

    fits = {}
    for method in ("log_ratio", "direct"):
        for w in L.WINDOWS:
            tr, te = L.split(panel, w)
            for s in ALL_SEEDS:
                t = time.time()
                fits[method, w, s] = L.lgb_fit(tr, te, cols, method, seed=s)[400]
                print(method, w, s, f"{L.score_pair(te, *fits[method, w, s])['mae']:.2f}", f"{time.time() - t:.1f}s", flush=True)

    te = {w: L.split(panel, w)[1] for w in L.WINDOWS}
    build = te["B"].building_key
    out = {}

    single = {m: pd.DataFrame([{w: L.score_pair(te[w], *fits[m, w, s])["mae"] for w in L.WINDOWS} | {
        "B_w100": L.score_pair(te["B"], *fits[m, "B", s])["within100"]} for s in ALL_SEEDS], index=ALL_SEEDS) for m in ("log_ratio", "direct")}
    for m, d in single.items():
        print(m, "\n", d.round(3).to_string(), "\n", d.agg(["mean", "std", "min", "max"]).round(3).to_string(), flush=True)
    out["single_seed"] = {m: d.to_dict() for m, d in single.items()}
    out["five_seed_sd"] = {m: dict(mean=float(d.B.iloc[:5].mean()), sd=float(d.B.iloc[:5].std()), lo=float(d.B.iloc[:5].min()), hi=float(d.B.iloc[:5].max()),
                                   w100_mean=float(d.B_w100.iloc[:5].mean()), w100_sd=float(d.B_w100.iloc[:5].std())) for m, d in single.items()}

    def ens(m, w, seeds):
        return tuple(np.mean([fits[m, w, s][i] for s in seeds], axis=0) for i in (0, 1))

    # Spread of a k-seed average, from random subsets of the ten seeds.
    rng = np.random.default_rng(0)
    spread = {}
    for k in (1, 3, 5, 10):
        subsets = [ALL_SEEDS] if k == 10 else [tuple(rng.choice(ALL_SEEDS, k, replace=False)) for _ in range(40)]
        v = [L.score_pair(te["B"], *ens("log_ratio", "B", s)) for s in subsets]
        spread[k] = dict(mean=float(np.mean([x["mae"] for x in v])), sd=float(np.std([x["mae"] for x in v])),
                         w100=float(np.mean([x["within100"] for x in v])))
    print("k-seed", json.dumps(spread, indent=0), flush=True)
    out["k_seed_spread"] = spread

    # Bootstrap of the single current model and the ten-seed average.
    base = L.abs_err(te["B"], *fits["log_ratio", "B", 7])
    out["bootstrap_rows"] = L.bootstrap_ci(base)
    out["bootstrap_buildings"] = L.bootstrap_ci(base, build)
    print("bootstrap rows", out["bootstrap_rows"], "\nbootstrap buildings", out["bootstrap_buildings"], flush=True)

    # Blends, always on the ten-seed averages, plus the five-seed recipe the submission would use.
    rows = []
    for a, seeds in itertools.product((1.0, 0.75, 0.5, 0.25, 0.0), (L.SEEDS, ALL_SEEDS)):
        blended = {w: tuple(a * x + (1 - a) * y for x, y in zip(ens("log_ratio", w, seeds), ens("direct", w, seeds))) for w in L.WINDOWS}
        sc = {w: L.score_pair(te[w], *blended[w]) for w in L.WINDOWS}
        name = f"blend_log{a:.2f}_seeds{len(seeds)}"
        row = dict(name=name, n_seeds=len(seeds), w_log=a, **L.summarise(sc))
        row["run_id"] = L.log_row(name, f"lgbm_l1:blend:log_ratio={a}:seeds={len(seeds)}", sc["B"], row)
        L.save_preds(name, blended)
        err = L.abs_err(te["B"], *blended["B"])
        row["vs_seed7"] = L.paired_bootstrap(base, err)
        row["vs_seed7_buildings"] = L.paired_bootstrap(base, err, build, reps=1000)
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != "run_id"}, flush=True)
    out["blends"] = rows
    (L.OUT / "step12.json").write_text(json.dumps(out, indent=1, default=str))
    np.savez_compressed(L.OUT / "seed7.npz", pw=fits["log_ratio", "B", 7][0], pwo=fits["log_ratio", "B", 7][1])


if __name__ == "__main__":
    main()
