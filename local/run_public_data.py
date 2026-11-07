"""Fold B test of public-data feature groups. Fold A is never scored here. Prints aggregates only.

Usage: python local/run_public_data.py PANEL.parquet [--seeds 5] [--final FINAL.parquet]
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src import external as E  # noqa: E402
from src import models as M  # noqa: E402
from src.features import FEATURE_COLS  # noqa: E402

G = E.GROUPS
ALL = [c for cols in G.values() for c in cols]
RUNS = {
    "baseline": [],
    "+gnaf_strata": G["gnaf_strata"],
    "+gnaf_site": G["gnaf_site"],
    "+gnaf_floor": G["gnaf_floor"],
    "+gnaf_created": G["gnaf_created"],
    "+osm": G["osm"],
    "+cadastre": G["cadastre"],
    "+census": G["census"],
    "+poi": G["poi"],
    "+strata+osm": G["gnaf_strata"] + G["osm"],
    "+all": ALL,
}


def run(panel: pd.DataFrame, extra: list[str], seeds: int) -> dict:
    """Seed-averaged Fold B scores, overall and for flats with and without an earlier advert."""
    cols = FEATURE_COLS + [c for c in extra if c in panel.columns]
    test = M.fold_rows(panel, "B")
    train = M.train_rows(panel, C.FOLDS["B"]["cutoff"], M.N_ORIGINS)
    first = test.anchor_own.isna()
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for s in range(seeds):
            cfg = dict(name=f"pd_{s}", frame=panel, feature_cols=cols, methods=["log_ratio"], seed=100 + s, log=False,
                       out_path=Path(tmp) / "r.json")
            res = M.run_experiment(cfg)
            assert not res["fold_a_reported"] and "A" not in res["folds"]
            model = M.fit_model(train, res["features"], "log_ratio", cfg)
            rows.append(dict(mae=res["folds"]["B"]["mae"], within100=res["folds"]["B"]["within100"], bias=res["folds"]["B"]["bias"],
                             first=M.evaluate(model, test[first])["mae"], repeat=M.evaluate(model, test[~first])["mae"]))
    d = pd.DataFrame(rows)
    return dict(n_cols=len(cols), mae=d.mae.mean(), sd=d.mae.std(), within100=d.within100.mean(), bias=d.bias.mean(),
                first_time=d["first"].mean(), repeat=d.repeat.mean(), n_first=int(first.sum()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("panel")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--final")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    raw = pd.read_parquet(a.panel)
    panel = E.attach(raw)
    panel.attrs["feature_cols"] = FEATURE_COLS
    print(f"panel {panel.shape}; Fold A rows are present in the frame but never scored")
    out = {}
    for name, extra in RUNS.items():
        if a.only and name not in a.only:
            continue
        out[name] = run(panel, extra, a.seeds)
        r = out[name]
        print(f"{name:16s} cols {r['n_cols']:3d}  FoldB {r['mae']:6.2f} sd {r['sd']:.2f}  w100 {r['within100']:.3f}  bias {r['bias']:+.2f}  "
              f"first-time {r['first_time']:6.2f} (n={r['n_first']})  repeat {r['repeat']:6.2f}", flush=True)
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (C.OUT_DIR / "public_data_foldB.json").write_text(json.dumps(out, indent=1))

    if a.final:
        fin = E.attach(pd.read_parquet(a.final), cutoff=C.AS_OF)
        cols = [c for c in ALL if c in panel.columns and c in fin.columns]
        print(E.blank_rates(M.fold_rows(panel, "B"), fin, cols).to_string())
        tr = M.train_rows(panel, C.FOLDS["B"]["cutoff"], M.N_ORIGINS)
        print("train rows blank, for comparison:\n", tr[cols].isna().mean().round(4).to_string())


if __name__ == "__main__":
    main()
