"""Write features_foldA.parquet and features_foldB.parquet to C.OUT_DIR for the Colab experiments.

Each file holds a fold's train and test rows with a `split` column. Train rows come from rolling
pseudo-cutoffs: each cutoff predicts the next two months, and every train target is on or before
the fold cutoff, so nothing in a fold's train set was unknowable at that cutoff.

Parquet metadata records the feature list and whether the data is mock, so the notebook can refuse
a mock file in a real run.

Run: PYTHONPATH=. python scripts/export_features.py [--origins 12]
"""
import argparse
import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas.tseries.offsets import MonthEnd

from src import config as C
from src import features as F
from src.load import fold_test_rows, load_input, load_leases

HORIZON_MONTHS = 2
ROW_COLS = ["gnaf_id", "target_date", "rent", "advert_car_space", "input_car_space", "car_space_resolved"]
# Used only until src.features.FEATURE_COLS lands. car_spaces is left out on purpose: it would
# contradict car_space_resolved in the with/without passes.
DEFAULT_FEATURES = ["bedrooms", "bathrooms", "floor_level", "flats_in_building", "block_new_homes_2021_2026",
                    "block_homes_2026", "ever_advertised_for_lease", "latitude", "longitude", "suburb",
                    "property_type", "base_with", "base_without", "attrs_age_days", "listed_age_days",
                    "car_space_resolved"]


def pseudo_cutoffs(cutoff, n_origins: int) -> list:
    """Month-ends two months apart, oldest first, so the target windows tile without overlap."""
    cutoff = pd.Timestamp(cutoff)
    return [cutoff + MonthEnd(-HORIZON_MONTHS * k) for k in range(n_origins, 0, -1)]


def window_rows(pseudo_cutoff) -> pd.DataFrame:
    """Adverts for input flats in the two months after a pseudo-cutoff."""
    end = pseudo_cutoff + MonthEnd(HORIZON_MONTHS)
    L = load_leases()
    t = L[(L.lease_date > pseudo_cutoff) & (L.lease_date <= end) & L.gnaf_id.isin(load_input().gnaf_id)]
    return t.assign(target_date=t.lease_date)[ROW_COLS].reset_index(drop=True)


def fold_frame(fold: str, n_origins: int) -> pd.DataFrame:
    cutoff = C.FOLDS[fold]["cutoff"]
    parts = []
    for p in pseudo_cutoffs(cutoff, n_origins):
        rows = window_rows(p)
        if len(rows):
            parts.append(F.build_features(p, rows).assign(split="train", pseudo_cutoff=p))
    parts.append(F.build_features(cutoff, fold_test_rows(fold)).assign(split="test", pseudo_cutoff=cutoff))
    df = pd.concat(parts, ignore_index=True)

    train, test = df[df.split == "train"], df[df.split == "test"]
    assert (df.target_date > df.pseudo_cutoff).all(), "a row can see its own advert"
    assert train.target_date.max() <= cutoff, "train target after the fold cutoff"
    assert test.target_date.between(C.FOLDS[fold]["start"], C.FOLDS[fold]["end"]).all(), "test row outside the fold"
    return df


def write_parquet(df: pd.DataFrame, path, feature_cols: list) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    meta = {**(table.schema.metadata or {}),
            b"flats_feature_cols": json.dumps(feature_cols).encode(),
            b"flats_source": b"mock" if C.MOCK else b"real"}
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.replace_schema_metadata(meta), path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origins", type=int, default=12, help="pseudo-cutoffs per fold; 12 covers two years")
    args = ap.parse_args()
    wanted = getattr(F, "FEATURE_COLS", DEFAULT_FEATURES)
    for fold in C.FOLDS:
        df = fold_frame(fold, args.origins)
        missing = [c for c in wanted if c not in df.columns]
        assert not missing, f"feature columns not built: {missing}"
        path = C.OUT_DIR / f"features_fold{fold}.parquet"
        write_parquet(df, path, list(wanted))
        print(f"Fold {fold}: {df.split.value_counts().to_dict()} -> {path}")


if __name__ == "__main__":
    main()
