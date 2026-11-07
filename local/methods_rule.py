"""Replay parking.pick_premium seed by seed on Fold B, to see how stable decision 15's margin is.

The rule compares two-pass against the within-building premium on the midpoint construction. The submission path
(compose_prices) keeps the flat's own pass instead, so both margins are printed.
"""
import json

import numpy as np
import pandas as pd

from local import methods_lib as L
from src import parking
from src import models as M


def main():
    panel = L.load_panel()
    cols = L.columns(panel)
    tr, te = L.split(panel, "B")
    wb = parking.premium_for(te, parking.estimate_premium(L.WINDOWS["B"]))
    rows, flips = [], None
    for s in L.SEEDS + (202, 303, 404, 505, 606):
        pw, pwo = L.lgb_fit(tr, te, cols, "log_ratio", seed=s)[400]
        two = te[["advert_car_space", "input_car_space", "rent"]].assign(rent_with_car_space=pw, rent_without_car_space=pwo)
        c = parking.compare_premium_methods(two, wb).set_index(["method", "segment"])
        model, mid = c.loc[("model", "flipped")], c.loc[("premium_mid", "flipped")]
        e_two, e_wb = L.abs_err(te, pw, pwo), L.abs_err(te, pw, pwo, wb)
        mask = (te.advert_car_space.notna() & te.input_car_space.notna() & ((te.advert_car_space > 0) != (te.input_car_space > 0))).reindex(e_two.index)
        rows.append(dict(seed=s, n=int(model.n), two_pass=float(model.mae), mid=float(mid.mae), rule_margin=float(mid.d_mae),
                         pick=parking.pick_premium(c.reset_index())[0], compose_wb=float(e_wb[mask].median()),
                         compose_margin=float(e_wb[mask].median() - e_two[mask].median()), model_gap=float(model.median_gap)))
        print(rows[-1], flush=True)
    d = pd.DataFrame(rows)
    print(d[["two_pass", "mid", "rule_margin", "compose_wb", "compose_margin", "model_gap"]].agg(["mean", "std", "min", "max"]).round(2).to_string())
    print(d.pick.value_counts().to_string())
    (L.OUT / "step7_rule.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
