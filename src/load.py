"""Load supplied files into a standard shape. AS_OF is enforced here, on every dated table."""
import re
from functools import lru_cache

import pandas as pd

from . import config as C

# The past-lease schema is not documented. Map likely names; fail loudly on anything else.
LEASE_ALIASES = {
    "lease_date": ["lease_date", "advert_date", "advertised_date", "listed_date", "date_listed", "listing_date",
                   "first_seen", "date", "start_date", "ad_date", "date_advertised", "first_listed", "lease_start"],
    "rent": ["rent", "weekly_rent", "rent_pw", "price", "price_pw", "asking_rent", "rent_per_week", "advertised_rent",
             "weekly_price", "rent_weekly"],
    "car_space": ["car_space", "parking", "has_parking", "has_car_space"],
    "car_spaces": ["car_spaces", "carspaces", "cars", "parking_spaces", "garage_spaces"],
    "bedrooms": ["bedrooms", "beds", "bed"],
    "bathrooms": ["bathrooms", "baths", "bath"],
}
RENT_MIN, RENT_MAX = 100, 10_000  # weekly dollars; outside this is a data error, not a rent


def parse_date(s: pd.Series) -> pd.Series:
    """Parse dates; the 0001-01-01 'unknown' sentinel becomes NaT."""
    s = s.astype("string").str.slice(0, 10).replace("0001-01-01", pd.NA)
    return pd.to_datetime(s, errors="coerce")


def norm_addr(s: pd.Series) -> pd.Series:
    return s.astype("string").str.upper().str.replace(r"[^A-Z0-9 ]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()


def _rename(df: pd.DataFrame, required=("lease_date", "rent")) -> pd.DataFrame:
    lower = {c.lower().strip(): c for c in df.columns}
    ren = {}
    for std, names in LEASE_ALIASES.items():
        hit = next((lower[n] for n in names if n in lower), None)
        if hit:
            ren[hit] = std
    df = df.rename(columns=ren)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Lease file: no column for {missing}. Columns are {list(df.columns)}. "
                         f"Add the real name to LEASE_ALIASES in src/load.py.")
    return df


@lru_cache(maxsize=1)
def load_input() -> pd.DataFrame:
    df = pd.read_csv(C.INPUT_CSV, dtype={"gnaf_id": "string", "mb": "string"}, low_memory=False)
    assert df.gnaf_id.is_unique, "input gnaf_id not unique"
    df["building_key"] = norm_addr(df.building_address)
    for c in ("first_ever_listed", "attrs_last_seen"):
        df[c] = parse_date(df[c])
    return df


@lru_cache(maxsize=1)
def load_leases() -> pd.DataFrame:
    """Leases dated <= AS_OF, tied to a flat where possible, with parking resolved."""
    raw = _rename(pd.read_csv(C.LEASES_CSV, low_memory=False))
    raw["lease_date"] = parse_date(raw["lease_date"])
    raw["rent"] = pd.to_numeric(raw["rent"], errors="coerce")
    df = raw[raw.lease_date.notna() & (raw.lease_date <= C.AS_OF) & raw.rent.between(RENT_MIN, RENT_MAX)].copy()

    flats = load_input()
    if "gnaf_id" in df.columns:
        df["gnaf_id"] = df["gnaf_id"].astype("string")
    elif "address" in df.columns:
        key = norm_addr(flats.address)
        df["gnaf_id"] = norm_addr(df.address).map(dict(zip(key[~key.duplicated()], flats.gnaf_id[~key.duplicated()])))
    else:
        df["gnaf_id"] = pd.NA
    if "building_address" in df.columns:
        df["building_key"] = norm_addr(df.building_address)
    else:
        df = df.merge(flats[["gnaf_id", "building_key"]], on="gnaf_id", how="left")

    if "car_space" not in df.columns:
        df["car_space"] = (pd.to_numeric(df["car_spaces"], errors="coerce") > 0).where(df["car_spaces"].notna()) \
            if "car_spaces" in df.columns else pd.NA
    df["advert_car_space"] = _to_flag(df["car_space"])

    # Flat attributes come from the input; the lease's own values win where it has them.
    extra = flats[["gnaf_id", "suburb", "bedrooms", "bathrooms", "car_space"]].rename(
        columns={"suburb": "suburb_f", "bedrooms": "bedrooms_f", "bathrooms": "bathrooms_f", "car_space": "input_car_space"})
    df = df.merge(extra, on="gnaf_id", how="left")
    b_sub = flats.drop_duplicates("building_key").set_index("building_key").suburb
    df["suburb"] = df["suburb_f"].fillna(df.building_key.map(b_sub)) if "suburb" not in df.columns \
        else df["suburb_f"].fillna(df["suburb"])
    df["bedrooms"] = pd.to_numeric(df.get("bedrooms"), errors="coerce").fillna(df["bedrooms_f"])
    df["bathrooms"] = pd.to_numeric(df.get("bathrooms"), errors="coerce").fillna(df["bathrooms_f"])
    df["car_space_resolved"] = df["advert_car_space"].fillna(df["input_car_space"])
    return df.drop(columns=["suburb_f", "bedrooms_f", "bathrooms_f"]).sort_values("lease_date").reset_index(drop=True)


def _to_flag(s: pd.Series) -> pd.Series:
    """Parking as 1.0/0.0/NaN from numbers, booleans or yes/no text."""
    t = s.astype("string").str.strip().str.lower()
    out = t.map({"1": 1.0, "1.0": 1.0, "true": 1.0, "yes": 1.0, "y": 1.0, "0": 0.0, "0.0": 0.0, "false": 0.0, "no": 0.0, "n": 0.0})
    num = pd.to_numeric(t, errors="coerce")
    return out.astype("float").fillna((num > 0).astype("float").where(num.notna()))


@lru_cache(maxsize=1)
def load_blocks() -> pd.DataFrame:
    return pd.read_csv(C.BLOCKS_CSV, dtype={"mb": "string"}, low_memory=False)


def fold_test_rows(fold: str) -> pd.DataFrame:
    """Adverts in a fold's test window for input flats, excluding only those with unknown parking."""
    f = C.FOLDS[fold]
    L = load_leases()
    t = L[L.lease_date.between(f["start"], f["end"]) & L.gnaf_id.isin(load_input().gnaf_id)]
    return t.assign(target_date=t.lease_date)[
        ["gnaf_id", "target_date", "rent", "advert_car_space", "input_car_space", "car_space_resolved"]
    ].reset_index(drop=True)
