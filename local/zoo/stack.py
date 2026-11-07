"""Score zoo members and choose a blend. Stage `select` never reads Fold A. Stage `final` reads it once and writes the submission.
usage: python local/zoo/stack.py select | final"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from local.zoo import lib as Z
from src import config as C, models as M
from src.validate import score, scored_prediction, log_experiment

pd.set_option("display.width", 180, "display.max_rows", 200)
DEV = ["w6", "w5", "w4", "w3", "w2", "w1", "B"]
CONTROL = "lgb_l1__base"
OUT = Z.ZOO / "selection.json"
r10 = lambda x: np.round(np.asarray(x, float) / 10) * 10


def read(windows):
    """{member: {window: (log_with, log_without)}} for members that have every window, aligned to the panel rows."""
    panel, final = Z.load()
    truth = {w: Z.split(panel, final, w)[1] for w in windows}
    res = {}
    for d in sorted(Z.PRED.glob("*__*")):
        if not all((d / f"{w}.parquet").exists() for w in windows + ["F"]):
            print("incomplete, skipped:", d.name); continue
        res[d.name] = {}
        for w in windows:
            p = pd.read_parquet(d / f"{w}.parquet")
            assert (p.gnaf_id.to_numpy() == truth[w].gnaf_id.astype(str).to_numpy()).all(), (d.name, w)
            res[d.name][w] = (np.log(p.pred_with.clip(lower=M.PRED_FLOOR).to_numpy()), np.log(p.pred_without.clip(lower=M.PRED_FLOOR).to_numpy()))
    return truth, res


def prices(lw, lwo, t, premium=None):
    """Same two-pass rule as the submission: own parking keeps its pass, the other column moves by the floored gap."""
    w, wo = np.exp(lw), np.exp(lwo)
    p = pd.DataFrame({"pred_with": w, "pred_without": wo, "premium": w - wo}, index=t.index)
    return M.compose_prices(p, t["input_car_space"], premium)


def quote10(c, car):
    """Round the column the flat is expected to be scored on to $10, and move the other by a gap rounded to $10."""
    car = pd.Series(np.asarray(car, float), index=c.index)
    gap = r10((c.rent_with_car_space - c.rent_without_car_space).clip(lower=0))
    w, wo = r10(c.rent_with_car_space), r10(c.rent_without_car_space)
    with_ = pd.Series(np.where(car > 0, w, wo + gap), index=c.index)
    without = pd.Series(np.where(car > 0, w - gap, wo), index=c.index).clip(lower=M.PRED_FLOOR)
    return pd.DataFrame({"rent_with_car_space": np.maximum(with_, without), "rent_without_car_space": without})


def frame(lw, lwo, t, rounded=False):
    c = prices(lw, lwo, t)
    if rounded:
        c = quote10(c, t["input_car_space"])
    return pd.concat([t[["rent", "advert_car_space", "input_car_space"]], c], axis=1)


def abs_err(lw, lwo, t, rounded=False):
    s = scored_prediction(frame(lw, lwo, t, rounded))
    return (s.pred - s.rent).abs().to_numpy()


def pooled(ws, truth, get, rounded=False):
    return np.concatenate([abs_err(*get(w), truth[w], rounded) for w in ws])


def obj(ae):
    """Clipped L1: the body of the errors, which is what a median cares about."""
    return np.minimum(ae, 120).mean()


def greedy(names, ws, truth, res, steps=25):
    """Caruana selection with replacement, in log space."""
    chosen, S = [], {w: None for w in ws}
    add = lambda w, n: tuple((a if S[w] is None else S[w][i] + a) for i, a in enumerate(res[n][w]))
    for _ in range(steps):
        k = len(chosen) + 1
        best, bv = None, np.inf
        for n in names:
            v = obj(np.concatenate([abs_err(*(x / k for x in add(w, n)), truth[w]) for w in ws]))
            if v < bv:
                best, bv = n, v
        S = {w: add(w, best) for w in ws}
        chosen.append(best)
    return pd.Series(chosen).value_counts(normalize=True).to_dict()


def mix(weights, res, w):
    return tuple(sum(a * res[n][w][i] for n, a in weights.items()) for i in (0, 1))


def boot(a, b, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    d = np.median(a[idx], axis=1) - np.median(b[idx], axis=1)
    return float(np.median(a) - np.median(b)), float(np.quantile(d, .025)), float(np.quantile(d, .975))


def select():
    truth, res = read(DEV)
    rows = []
    for n in res:
        g = lambda w: res[n][w]
        e = pooled(DEV[:6], truth, g); b = abs_err(*g("B"), truth["B"])
        rows.append(dict(member=n, med_pre=round(np.median(e), 2), w100=round((e <= 100).mean(), 4), clip=round(obj(e), 2), B=round(np.median(b), 2),
                         r10_pre=np.median(pooled(DEV[:6], truth, g, True)), r10_B=np.median(abs_err(*g("B"), truth["B"], True))))
    board = pd.DataFrame(rows).sort_values("clip"); print(board.to_string(index=False))

    # Residual diversity on the pre-B windows
    top = board.member.head(8).tolist()
    E = pd.DataFrame({n: np.concatenate([res[n][w][1] - np.log(truth[w].rent.to_numpy(float)) for w in DEV[:6]]) for n in top})
    print("\nerror correlation, top 8\n", E.corr(method="spearman").round(2).to_string())

    # Forward chaining: weights only ever see earlier windows
    names = list(res)
    fwd = {"control": {}, "mean_raw": {}, "mean_top5": {}, "greedy": {}}
    tests = DEV[2:]
    for i, t in enumerate(tests):
        past = DEV[:DEV.index(t)]
        sc = {n: obj(pooled(past, truth, lambda w, n=n: res[n][w])) for n in names}
        top5 = sorted(sc, key=sc.get)[:5]
        cand = {"control": {CONTROL: 1.0}, "mean_raw": {n: 1 / sum(x.endswith("__raw") for x in names) for n in names if n.endswith("__raw")},
                "mean_top5": {n: .2 for n in top5}, "greedy": greedy(names, past, truth, res)}
        for k, wts in cand.items():
            fwd[k][t] = mix(wts, res, t)
        print("forward", t, "greedy", {k: round(v, 2) for k, v in cand["greedy"].items()}, flush=True)
    base = pooled(tests, truth, lambda w: fwd["control"][w])
    print("\nforward windows", tests, "n", len(base))
    for k in fwd:
        e = pooled(tests, truth, lambda w: fwd[k][w]); er = pooled(tests, truth, lambda w: fwd[k][w], True)
        d = boot(e, base)
        print(f"{k:10s} raw {np.median(e):.2f} clip {obj(e):.2f} w100 {(e <= 100).mean():.4f} diff vs control {d[0]:+.2f} [{d[1]:+.2f},{d[2]:+.2f}] | $10 {np.median(er):.0f} "
              f"| per window raw {[round(float(np.median(abs_err(*fwd[k][w], truth[w]))), 1) for w in tests]} $10 {[float(np.median(abs_err(*fwd[k][w], truth[w], True))) for w in tests]}")
    wts = greedy(names, DEV, truth, res)
    print("\nfinal greedy weights on all dev windows", {k: round(v, 2) for k, v in wts.items()})
    M.write_json(dict(greedy_all=wts, top5=board.member.head(5).tolist()), OUT)


def choose(sel, method):
    return sel["greedy_all"] if method == "greedy" else {n: 1 / len(sel["top5"]) for n in sel["top5"]}


def premium_rule(wts, truth, res):
    """Decision 15's fixed rule, rerun for this blend on Fold B."""
    from src import parking
    t = truth["B"]
    lw, lwo = mix(wts, res, "B")
    two = t[["advert_car_space", "input_car_space", "rent"]].assign(rent_with_car_space=np.exp(lw), rent_without_car_space=np.exp(lwo))
    table = parking.premium_for(t, parking.estimate_premium(C.FOLDS["B"]["cutoff"]))
    cmp_ = parking.compare_premium_methods(two, table)
    print(cmp_[cmp_.method.isin(["model", "premium_mid"])][["method", "segment", "n", "mae", "d_mae", "median_gap"]].round(2).to_string(index=False))
    source, rule = parking.pick_premium(cmp_)
    print("premium source:", source, "|", rule, flush=True)
    return source


def write(method):
    """Fold A is not read here."""
    from src import parking
    from src.predict import write_submission
    wts = choose(json.loads(OUT.read_text()), method)
    truth, res = read(["B"])
    source = premium_rule(wts, truth, res)
    _, X = Z.load()
    F = {n: pd.read_parquet(Z.PRED / n / "F.parquet") for n in wts}
    assert all((f.gnaf_id.to_numpy() == X.gnaf_id.astype(str).to_numpy()).all() for f in F.values())
    lw = sum(a * np.log(F[n].pred_with.clip(lower=M.PRED_FLOOR).to_numpy()) for n, a in wts.items())
    lwo = sum(a * np.log(F[n].pred_without.clip(lower=M.PRED_FLOOR).to_numpy()) for n, a in wts.items())
    prem = parking.premium_for(X, parking.estimate_premium(C.AS_OF)) if source == "within_building" else None
    c = quote10(prices(lw, lwo, X, prem), X.input_car_space)
    assert c.notna().all().all()
    sub = write_submission(pd.concat([X[["gnaf_id"]].astype({"gnaf_id": object}), c], axis=1))
    gap = sub.rent_with_car_space - sub.rent_without_car_space
    print("rows", len(sub), "gap median", gap.median(), "mean", round(gap.mean(), 1), "zero gap", round((gap == 0).mean(), 3), "median without", sub.rent_without_car_space.median())
    print("wrote", C.SUBMISSION_CSV)
    M.write_json(dict(method=method, weights=wts, premium_source=source), Z.ZOO / "written.json")


def fold_a(method):
    """The single read of Fold A."""
    from src import parking
    wts = choose(json.loads(OUT.read_text()), method)
    source = json.loads((Z.ZOO / "written.json").read_text())["premium_source"]
    truth, res = read(["B", "A"])
    got = {}
    for label, ww in (("control", {CONTROL: 1.0}), ("blend", wts)):
        for w in ("B", "A"):
            t = truth[w]
            prem = parking.premium_for(t, parking.estimate_premium(C.FOLDS[w]["cutoff"])) if source == "within_building" and label == "blend" else None
            c = prices(*mix(ww, res, w), t, prem)
            base = t[["rent", "advert_car_space", "input_car_space"]]
            got[label, w, "raw"] = score(pd.concat([base, c], axis=1))
            got[label, w, "r10"] = score(pd.concat([base, quote10(c, t["input_car_space"])], axis=1))
            print(label, w, "raw", {k: round(v, 3) for k, v in got[label, w, "raw"].items() if k in ("n", "mae", "within100", "r2", "bias")},
                  "| $10", {k: round(v, 3) for k, v in got[label, w, "r10"].items() if k in ("mae", "within100", "r2")})
    feats = f"raw last rent and v2 history features, {len(wts)} members"
    for tag, key in (("raw", "raw"), ("rounded to $10", "r10")):
        print("logged", log_experiment(f"Zoo blend, {method} weights fitted on six windows and Fold B, {tag}", feats, f"zoo:{method}", got["blend", "A", key], got["blend", "B", key]))
    M.write_json({f"{a}_{b}_{c}": v for (a, b, c), v in got.items()}, Z.ZOO / "fold_a_result.json")
    return got


if __name__ == "__main__":
    a = sys.argv[1]; m = sys.argv[2] if len(sys.argv) > 2 else "greedy"
    {"select": lambda: select(), "write": lambda: write(m), "fold_a": lambda: fold_a(m)}[a]()
