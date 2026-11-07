"""Poison test for the new features, and rounding outcomes for the forward blend. Fold A is not read."""
import sys, json
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from local.zoo import lib as Z, stack as S
from src.load import load_leases
L = load_leases()
p, f = Z.load()
cut = pd.Timestamp("2025-09-30")
rows = p[p.origin == cut].drop(columns=Z.RAW_COLS + Z.V2_COLS)
clean = Z.add_v2(Z.add_raw(rows, L), L)
bad = L.copy(); late = bad.lease_date > cut
bad.loc[late, "rent"] *= 5; bad.loc[late, "bedrooms"] = 9
pois = Z.add_v2(Z.add_raw(rows, bad), bad)
moved = {c: int(((clean[c] != pois[c]) & ~(clean[c].isna() & pois[c].isna())).sum()) for c in Z.RAW_COLS + Z.V2_COLS}
print("poison after", cut.date(), "rows", len(rows), "feature values moved:", moved)
fw = p[p.origin == pd.Timestamp("2026-03-31")]
print("blank rates, Fold B rows vs final rows:", {c: (round(fw[c].isna().mean(), 3), round(f[c].isna().mean(), 3)) for c in Z.RAW_COLS + Z.V2_COLS})
truth, res = S.read(S.DEV)
for label, path in (("raw pool", "selection_raw.json"), ("v2 pool", "selection.json")):
    wts = json.loads((Z.ZOO / path).read_text())["greedy_all"]
    print(label, "(in-sample weights) window: raw | $10 | $5-offset | unrounded<=35 share")
    for w in S.DEV:
        t = truth[w]; c = S.prices(*S.mix(wts, res, w), t)
        sc = S.scored_prediction(pd.concat([t[["rent", "advert_car_space", "input_car_space"]], c], axis=1))
        y, q = sc.rent.to_numpy(float), sc.pred.to_numpy(float)
        print(" ", w, round(np.median(abs(q - y)), 1), np.median(abs(np.round(q / 10) * 10 - y)), np.median(abs(np.round((q - 5) / 10) * 10 + 5 - y)), round(float((abs(q - y) <= 35).mean()), 3))
