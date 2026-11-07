"""Fold B confirmation of the raw last rent features, 5-seed averaged. Fold A rows are never scored here."""
import json

import numpy as np
import pandas as pd

from src import config as C
from src import features as F
from src import models as M
from src.predict import build_panel, final_rows
from src.validate import scored_prediction, score

panel = build_panel(M.N_ORIGINS)
cut = C.FOLDS["B"]["cutoff"]
train, test = M.train_rows(panel, cut, M.N_ORIGINS), M.fold_rows(panel, "B")
fin = final_rows()
print("blank rate fold vs final:", {c: (round(test[c].isna().mean(), 3), round(fin[c].isna().mean(), 3)) for c in F.RAW_PREV_COLS})
base = [c for c in F.FEATURE_COLS if c not in F.RAW_PREV_COLS]
err, res = {}, {}
for name, cols in (("without", base), ("with", F.FEATURE_COLS)):
    m = M.fit_model(train, cols, "log_ratio", dict(seeds=M.SEEDS))
    prices = M.compose_prices(M.predict_two_pass(m, test), test.input_car_space)
    d = pd.concat([test[["rent", "advert_car_space", "input_car_space"]], prices], axis=1)
    res[name] = {k: round(v, 3) for k, v in score(d).items()}
    s = scored_prediction(d)
    err[name] = (s.pred - s.rent).abs()
    first = test.anchor_own.isna().reindex(s.index)
    res[name]["first_time"] = round(err[name][first].median(), 2)
    res[name]["repeat"] = round(err[name][~first].median(), 2)
rng = np.random.default_rng(1)
a, b = err["without"].to_numpy(), err["with"].to_numpy()
d = [np.median(b[i]) - np.median(a[i]) for i in (rng.integers(0, len(a), len(a)) for _ in range(2000))]
res["gap_95"] = [round(float(x), 2) for x in np.percentile(d, [2.5, 50, 97.5])]
print(json.dumps(res, indent=1))
