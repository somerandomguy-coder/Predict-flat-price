"""Does TabM add anything? Pre-B windows only."""
import sys, json
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from local.zoo import lib as Z, stack as S
from src import models as M
ws = S.DEV[:6]
panel, final = Z.load()
truth = {w: Z.split(panel, final, w)[1] for w in ws}
wts = json.loads((Z.ZOO / "selection_v2.json").read_text())["greedy_all"]
def lg(n, w):
    p = pd.read_parquet(Z.PRED / n / f"{w}.parquet")
    return np.log(p.pred_with.clip(lower=100).to_numpy()), np.log(p.pred_without.clip(lower=100).to_numpy())
def blend(w):
    a = [lg(n, w) for n in wts]; k = list(wts.values())
    return tuple(sum(x * p[i] for x, p in zip(k, a)) for i in (0, 1))
ae = lambda g: np.concatenate([S.abs_err(*g(w), truth[w]) for w in ws])
b = ae(blend); t = ae(lambda w: lg("tabm__v2", w))
print("blend raw", round(np.median(b), 2), "clip", round(S.obj(b), 2), "| tabm raw", round(np.median(t), 2), "clip", round(S.obj(t), 2), "w100", round((t <= 100).mean(), 4))
for a in (.1, .2, .3):
    e = ae(lambda w: tuple((1 - a) * x + a * y for x, y in zip(blend(w), lg("tabm__v2", w))))
    d = S.boot(e, b, n=400)
    print(f"blend + {a:.0%} TabM: raw {np.median(e):.2f} clip {S.obj(e):.2f} w100 {(e <= 100).mean():.4f} diff {d[0]:+.2f} [{d[1]:+.2f},{d[2]:+.2f}]")
