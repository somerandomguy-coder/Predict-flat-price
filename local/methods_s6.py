"""Step 6: shrinking toward best_anchor, first-time flats, and a bias carried from the previous window.

First-time means anchor_own is blank. Everything is seed-averaged and scored on the usual windows, and first-time
and repeat flats are reported apart, because a change aimed at 25% of adverts barely moves the overall median.
"""
import json
import sys

import numpy as np
import pandas as pd

from local import methods_lib as L

SEEDS = L.SEEDS


def ensemble(tr, te, cols, method, params, rounds, weight=None):
    fits = [L.lgb_fit(tr, te, cols, method, params, rounds, s, weight)[rounds] for s in SEEDS]
    return tuple(np.mean([f[i] for f in fits], axis=0) for i in (0, 1))


def report(panel, name, preds, log=True) -> dict:
    sc, first, repeat = {}, {}, {}
    for w in preds:
        te = L.split(panel, w)[1]
        sc[w] = L.score_pair(te, *preds[w])
        err = L.abs_err(te, *preds[w])
        is_first = te.anchor_own.reindex(err.index).isna()
        first[w], repeat[w] = err[is_first], err[~is_first]
    row = dict(name=name, **L.summarise(sc), B_first_n=len(first["B"]), B_first=float(first["B"].median()), B_first_w100=float((first["B"] <= 100).mean()),
               B_repeat=float(repeat["B"].median()), B_repeat_w100=float((repeat["B"] <= 100).mean()),
               I_first=float(np.mean([first[w].median() for w in L.INNER if w in first])) if len(preds) > 1 else float("nan"),
               I_repeat=float(np.mean([repeat[w].median() for w in L.INNER if w in repeat])) if len(preds) > 1 else float("nan"))
    if log:
        row["run_id"] = L.log_row(name, f"lgbm_l1:step6:{name}", sc["B"], row)
    L.append_results("step6", [row])
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items() if k != "run_id"}, flush=True)
    return row


def main():
    params = eval(sys.argv[1]) if len(sys.argv) > 1 else {}
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    panel = L.load_panel()
    cols = L.columns(panel)

    joint, direct, only_first, only_first_direct, up2, up4 = ({} for _ in range(6))
    for w in L.WINDOWS:
        tr, te = L.split(panel, w)
        joint[w] = ensemble(tr, te, cols, "log_ratio", params, rounds)
        direct[w] = ensemble(tr, te, cols, "direct", params, rounds)
        first_tr = tr[tr.anchor_own.isna()]
        only_first[w] = ensemble(first_tr, te, cols, "log_ratio", params, rounds)
        only_first_direct[w] = ensemble(first_tr, te, cols, "direct", params, rounds)
        is_first = tr.anchor_own.isna().to_numpy()
        up2[w] = ensemble(tr, te, cols, "log_ratio", params, rounds, np.where(is_first, 2.0, 1.0))
        up4[w] = ensemble(tr, te, cols, "log_ratio", params, rounds, np.where(is_first, 4.0, 1.0))
        print(w, "first-time training rows", len(first_tr), "of", len(tr), flush=True)
    report(panel, "s6_joint", joint)

    def anchor(w):
        return L.split(panel, w)[1].best_anchor.to_numpy(float)

    for lam in (0.7, 0.8, 0.9, 0.95, 1.05, 1.1):
        report(panel, f"s6_shrink{lam}", {w: tuple(anchor(w) * (p / anchor(w)) ** lam for p in joint[w]) for w in L.WINDOWS})

    def route(alt, label):
        out = {}
        for w in L.WINDOWS:
            f = L.split(panel, w)[1].anchor_own.isna().to_numpy()
            out[w] = tuple(np.where(f, a, j) for a, j in zip(alt[w], joint[w]))
        report(panel, label, out)

    route(only_first, "s6_first_own_model")
    route(only_first_direct, "s6_first_own_direct")
    route(direct, "s6_first_joint_direct")
    route({w: tuple((a + b) / 2 for a, b in zip(direct[w], joint[w])) for w in L.WINDOWS}, "s6_first_half_direct")
    route({w: tuple((a + b) / 2 for a, b in zip(only_first[w], joint[w])) for w in L.WINDOWS}, "s6_first_half_own_model")
    report(panel, "s6_first_weight2", up2)
    report(panel, "s6_first_weight4", up4)

    # Bias carried forward: each window is corrected by the bias of the window just before it, whose labels end at its cutoff.
    order = ["I3", "I2", "I1", "B"]
    logbias = {}
    for w in order:
        te = L.split(panel, w)[1]
        s = L.scored_prediction(L.frame(te, *joint[w])) if hasattr(L, "scored_prediction") else None
        logbias[w] = float(np.median(np.log(s["pred"] / s["rent"])))
    print("log bias by window", {k: round(v, 4) for k, v in logbias.items()}, flush=True)
    for alpha in (0.5, 1.0):
        out = {w: tuple(p * np.exp(-alpha * logbias[prev]) for p in joint[w]) for prev, w in zip(order, order[1:])}
        row = report(panel, f"s6_carry_bias{alpha}", out)
        print("   same windows uncorrected:", {w: round(L.score_pair(L.split(panel, w)[1], *joint[w])["mae"], 2) for w in out}, flush=True)
    (L.OUT / "step6_logbias.json").write_text(json.dumps(logbias))


if __name__ == "__main__":
    main()
