"""Step 4: how many origins to train on, and recency weighting of training rows.

Weights use only a row's origin relative to the fold cutoff, so nothing later than the cutoff is touched.
The lease file changes around 2023 (origin 2023-03 has 628 adverts, 2023-05 has 1,847), so rows before the
break are also down-weighted or dropped on their own.
"""
import sys

import numpy as np
import pandas as pd

from local import methods_lib as L

BREAK = pd.Timestamp("2023-05-31")


def half_life(months):
    def fn(tr, cutoff):
        back = (cutoff.year - tr.origin.dt.year) * 12 + cutoff.month - tr.origin.dt.month
        return (0.5 ** (back.to_numpy(float) / months))
    return fn


def pre_break(weight):
    def fn(tr, cutoff):
        return np.where(tr.origin < BREAK, weight, 1.0)
    return fn


def main():
    params = eval(sys.argv[1]) if len(sys.argv) > 1 else {}
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    panel = L.load_panel()
    runs = [(f"s4_n{n}", n, None) for n in (4, 6, 8, 10, 12, 15, 18, 21, 24)]
    runs += [(f"s4_n{n}_hl{h}", n, half_life(h)) for n, h in ((12, 12), (12, 24), (18, 12), (18, 24), (24, 12), (24, 24))]
    runs += [(f"s4_n24_prebreak{w}", 24, pre_break(w)) for w in (0.25, 0.5)]
    for name, n, weight in runs:
        rows = L.run_lgb(panel, name, params=params, rounds=rounds, seeds=L.SEEDS, n=n, weight_fn=weight)
        L.append_results("step4", rows)


if __name__ == "__main__":
    main()
