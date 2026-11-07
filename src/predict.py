"""Build the rolling-origin panel, and turn per-flat predictions into a checked submission file."""
import importlib
import importlib.util

import pandas as pd

from . import config as C
from . import features
from . import models as M
from .load import _to_flag, load_input, load_leases
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


def window_rows(start, end) -> pd.DataFrame:
    """Adverts for input flats dated start..end inclusive, shaped like fold_test_rows."""
    L = load_leases()
    t = L[L.lease_date.between(start, end) & L.gnaf_id.isin(load_input().gnaf_id)]
    return t.assign(target_date=t.lease_date)[
        ["gnaf_id", "target_date", "rent", "advert_car_space", "input_car_space", "car_space_resolved"]
    ].reset_index(drop=True)


def feature_cols() -> list | None:
    """src.features.FEATURE_COLS once the anchors merge, else None, which selects the shim in src.models."""
    return getattr(features, "FEATURE_COLS", None)


def suburb_median(origin, X: pd.DataFrame) -> pd.Series:
    """12-month suburb median at origin. It is the report's simplest comparison, not a model input."""
    h = features.history_upto(origin)
    recent = h[h.lease_date > origin - pd.DateOffset(months=12)].dropna(subset=["suburb"])
    return features.hier_median(recent, X, levels=[["suburb"], []])


def build_panel(n_origins: int = M.N_ORIGINS) -> pd.DataFrame:
    """Every rolling-origin row needed by Fold A, Fold B and the final fit, in one frame.

    Origins are the month ends stepping back from AS_OF. The final fit takes the first n, Fold A
    (with its test rows) the next, Fold B the next, so n + 2 cover all three.
    """
    parts = []
    for origin in M.origins_for(C.AS_OF, n_origins + 2):
        rows = window_rows(origin + pd.Timedelta(days=1), origin + pd.offsets.MonthEnd(M.STEP_MONTHS))
        if len(rows):
            X = features.build_features(origin, rows)
            parts.append(X.assign(origin=origin, base_suburb=suburb_median(origin, X)))
    panel = pd.concat(parts, ignore_index=True).assign(data_mode="mock" if C.MOCK else "real")
    panel.attrs["feature_cols"] = feature_cols()
    return panel


def final_rows() -> pd.DataFrame:
    """Features for all input flats at AS_OF, aimed at the middle of the scored window."""
    flats = features.flats_asof(C.AS_OF)
    car = _to_flag(flats.car_space)
    rows = flats[["gnaf_id"]].assign(target_date=C.SCORE_MID, input_car_space=car, car_space_resolved=car)
    return M.ensure_best_anchor(features.build_features(C.AS_OF, rows))


def load_premium_table() -> tuple[pd.DataFrame | None, str]:
    """The parking lane's table if there is one: src.parking.premium_table(), else a file under OUT_DIR.

    Needs a `premium` column keyed by gnaf_id, or by any of building_key, suburb, bedrooms.
    """
    if importlib.util.find_spec("src.parking"):  # a broken module should fail here, not silently use two-pass
        parking = importlib.import_module("src.parking")
        if hasattr(parking, "premium_table"):
            return parking.premium_table(), "src.parking.premium_table"
    for path, read in ((C.OUT_DIR / "parking_premium.parquet", pd.read_parquet), (C.OUT_DIR / "parking_premium.csv", pd.read_csv)):
        if path.exists():
            return read(path), path.name
    return None, "two_pass"


def premium_for(rows: pd.DataFrame, table: pd.DataFrame | None) -> pd.Series | None:
    """The table's premium for each row, NaN where it has none. None when there is no table."""
    if table is None:
        return None
    keys = ["gnaf_id"] if "gnaf_id" in table.columns else [c for c in ("building_key", "suburb", "bedrooms") if c in table.columns]
    if not keys or "premium" not in table.columns:
        raise ValueError(f"premium table needs a key and a premium column, got {list(table.columns)}")
    hit = rows[keys].merge(table[keys + ["premium"]], on=keys, how="left", validate="m:1")["premium"]
    hit.index = rows.index
    return hit
