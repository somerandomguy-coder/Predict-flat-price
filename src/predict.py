"""Turn per-flat predictions into a checked submission file."""
import pandas as pd

from . import config as C
from .load import load_input
from .validate import check_submission, round_half


def write_submission(pred: pd.DataFrame, fallback: pd.DataFrame | None = None) -> pd.DataFrame:
    """pred: gnaf_id, rent_with_car_space, rent_without_car_space. Blanks come from fallback, then the median."""
    ids = load_input().gnaf_id
    sub = pd.DataFrame({"gnaf_id": ids}).merge(pred, on="gnaf_id", how="left")
    cols = ["rent_with_car_space", "rent_without_car_space"]
    if fallback is not None:
        fb = sub[["gnaf_id"]].merge(fallback, on="gnaf_id", how="left")
        for c in cols:
            sub[c] = sub[c].fillna(fb[c])
    for c in cols:
        sub[c] = sub[c].fillna(sub[c].median())
    sub["rent_without_car_space"] = round_half(sub.rent_without_car_space)
    sub["rent_with_car_space"] = round_half(sub.rent_with_car_space)
    sub["rent_with_car_space"] = sub[cols].max(axis=1)
    sub = sub[["gnaf_id"] + cols]
    check_submission(sub, ids)
    C.SUBMISSION_CSV.parent.mkdir(parents=True, exist_ok=True)
    tmp = C.SUBMISSION_CSV.with_suffix(".tmp")
    sub.to_csv(tmp, index=False, encoding="utf-8")
    tmp.replace(C.SUBMISSION_CSV)  # atomic, so a valid file is always on disk
    return sub
