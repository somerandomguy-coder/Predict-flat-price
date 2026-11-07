"""Figures for the recommended config from saved predictions: scores, intervals, and the premium rule on the five-seed model."""
import numpy as np
import pandas as pd

from local import methods_lib as L
from src import parking


def main():
    panel = L.load_panel()
    p = L.load_preds("blend_log1.00_seeds5")
    for w in L.WINDOWS:
        te = L.split(panel, w)[1]
        print(w, {k: round(v, 4) if isinstance(v, float) else v for k, v in L.score_pair(te, *p[w]).items()})
    te = L.split(panel, "B")[1]
    e = L.abs_err(te, *p["B"])
    print("rows", L.bootstrap_ci(e))
    print("buildings", L.bootstrap_ci(e, te.building_key))
    s7 = np.load(L.OUT / "seed7.npz")
    print("vs seed 7", L.paired_bootstrap(L.abs_err(te, s7["pw"], s7["pwo"]), e))

    wb = parking.premium_for(te, parking.estimate_premium(L.WINDOWS["B"]))
    pw, pwo = p["B"]
    two = te[["advert_car_space", "input_car_space", "rent"]].assign(rent_with_car_space=pw, rent_without_car_space=pwo)
    comparison = parking.compare_premium_methods(two, wb)
    print(comparison[comparison.segment == "flipped"][["method", "n", "mae", "d_mae", "median_gap"]].round(2).to_string(index=False))
    print("rule on five-seed model:", parking.pick_premium(comparison))

    adv, inp = te.advert_car_space.astype(float), te.input_car_space.astype(float)
    flip = adv.notna() & inp.notna() & ((adv > 0) != (inp > 0))
    gained, lost = flip & (adv > 0), flip & (adv == 0)
    gap = pd.Series(pw - pwo, index=te.index).clip(lower=0)
    for label, prem in (("two_pass", None), ("half", 0.5 * gap + 0.5 * wb), ("within_building", wb)):
        err = L.abs_err(te, pw, pwo, prem)
        print(label, "flipped", round(err[flip.reindex(err.index)].median(), 2), "| gained parking", int(gained.sum()), round(err[gained.reindex(err.index)].median(), 2),
              "| lost parking", int(lost.sum()), round(err[lost.reindex(err.index)].median(), 2), "| all", round(err.median(), 2), round((err <= 100).mean(), 4),
              "| median gap", round(float((gap if prem is None else prem).median()), 2))


if __name__ == "__main__":
    main()
