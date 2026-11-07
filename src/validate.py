"""Metrics exactly as the hackathon scores them, plus submission and experiment-log checks."""
import csv
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config as C

EXPERIMENT_COLS = ["run_id", "timestamp", "description", "features", "model", "foldA_mae", "foldA_within100",
                   "foldA_r2", "foldB_mae", "bias"]  # *_mae = MEDIAN absolute error, the rank metric


def scored_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """Pick the column matching advert parking, else input parking. Drop only when both are unknown."""
    park = df["advert_car_space"].astype(float).fillna(df["input_car_space"].astype(float))
    out = df[park.notna()].copy()
    p = park[park.notna()]
    out["pred"] = np.where(p > 0, out["rent_with_car_space"], out["rent_without_car_space"])
    return out


def score(df: pd.DataFrame) -> dict:
    """df needs: rent, rent_with_car_space, rent_without_car_space, advert_car_space, input_car_space.

    A blank scored prediction counts as an infinite error, worse than any answered miss.
    """
    s = scored_prediction(df)
    err = (s["pred"] - s["rent"]).astype(float)
    abs_err = err.abs().fillna(np.inf)
    ok = err.notna()
    sse = (err[ok] ** 2).sum()
    sst = ((s.loc[ok, "rent"] - s.loc[ok, "rent"].mean()) ** 2).sum()
    return dict(n=len(s), n_excluded=len(df) - len(s), n_blank=int((~ok).sum()), mae=float(abs_err.median()),
                within100=float((abs_err <= 100).mean()), r2=float(1 - sse / sst) if sst else float("nan"),
                bias=float(err.median()))


def round_half(x) -> np.ndarray:
    return np.round(np.asarray(x, dtype=float) * 2) / 2


def check_submission(sub: pd.DataFrame, input_ids: pd.Series) -> None:
    assert list(sub.columns) == ["gnaf_id", "rent_with_car_space", "rent_without_car_space"], list(sub.columns)
    assert sub.gnaf_id.is_unique, "duplicate gnaf_id"
    assert set(sub.gnaf_id) == set(input_ids), "ids differ from input"
    assert len(sub) == len(input_ids)
    w, wo = sub.rent_with_car_space, sub.rent_without_car_space
    assert w.notna().all() and wo.notna().all(), "blank rent"
    assert np.isfinite(w).all() and np.isfinite(wo).all(), "non-finite rent"
    assert (wo > 0).all(), "non-positive rent"
    assert (w >= wo).all(), "with < without"
    assert ((w * 2) % 1 == 0).all() and ((wo * 2) % 1 == 0).all(), "not whole or .50"


def log_experiment(description: str, features: str, model: str, a: dict, b: dict) -> str:
    """Append one row. Never rewrites the file, so history is preserved."""
    path = C.EXPERIMENTS_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:8]
    row = [run_id, datetime.now(timezone.utc).isoformat(timespec="seconds"), description, features, model,
           round(a["mae"], 2), round(a["within100"], 4), round(a["r2"], 4), round(b["mae"], 2), round(a["bias"], 2)]
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(EXPERIMENT_COLS)
        w.writerow(row)
    return run_id
