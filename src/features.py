"""build_features(cutoff_date, target_rows): the one place features are made.

Nothing dated after cutoff_date may reach a feature. That includes input columns that
carry later information: ever_advertised_for_lease, attrs_last_seen, first_ever_listed.
"""
import pandas as pd

from . import config as C
from .load import load_input, load_leases

BASELINE_LEVELS = [["suburb", "bedrooms", "car_space_resolved"], ["suburb", "bedrooms"], ["bedrooms"], []]
MIN_GROUP = 5


def history_upto(cutoff_date) -> pd.DataFrame:
    cutoff = pd.Timestamp(cutoff_date)
    assert cutoff <= C.AS_OF, f"cutoff {cutoff} is after the as-of date"
    h = load_leases()
    h = h[h.lease_date <= cutoff]
    assert h.lease_date.max() <= cutoff
    return h


def flats_asof(cutoff_date) -> pd.DataFrame:
    """Input flats as they were knowable at cutoff_date."""
    cutoff = pd.Timestamp(cutoff_date)
    f = load_input().copy()
    f["attrs_last_seen"] = f.attrs_last_seen.where(f.attrs_last_seen <= cutoff)
    f["first_ever_listed"] = f.first_ever_listed.where(f.first_ever_listed <= cutoff)
    f["ever_advertised_for_lease"] = f.gnaf_id.isin(history_upto(cutoff).gnaf_id.dropna()).astype("int8")
    return f


def hier_median(hist: pd.DataFrame, target: pd.DataFrame, levels=BASELINE_LEVELS, min_n=MIN_GROUP) -> pd.Series:
    """Median rent from the most specific group with at least min_n leases."""
    out = pd.Series(float("nan"), index=target.index)
    for keys in levels:
        if not keys:
            out = out.fillna(hist.rent.median())
            continue
        g = hist.groupby(keys, dropna=True).rent.agg(["median", "size"])
        g = g.loc[g["size"] >= min_n, "median"].rename("m").reset_index()
        m = target[keys].merge(g, on=keys, how="left")["m"]
        m.index = target.index
        out = out.fillna(m)
    return out


def build_features(cutoff_date, target_rows: pd.DataFrame) -> pd.DataFrame:
    """target_rows needs gnaf_id, target_date, car_space_resolved. Returns one row per target row."""
    cutoff = pd.Timestamp(cutoff_date)
    assert (target_rows.target_date > cutoff).all(), "targets must be after the cutoff"
    hist = history_upto(cutoff)
    flats = flats_asof(cutoff)
    X = target_rows.reset_index(drop=True).merge(
        flats.drop(columns=["address", "building_address", "car_space"]), on="gnaf_id", how="left", validate="m:1")

    recent = hist[hist.lease_date > cutoff - pd.DateOffset(months=12)]
    recent = recent.dropna(subset=["suburb", "bedrooms"])
    for name, park in (("base_with", 1.0), ("base_without", 0.0)):
        X[name] = hier_median(recent, X.assign(car_space_resolved=park))
    X["attrs_age_days"] = (cutoff - X.attrs_last_seen).dt.days
    X["listed_age_days"] = (cutoff - X.first_ever_listed).dt.days
    return X
