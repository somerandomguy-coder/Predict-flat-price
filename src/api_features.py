"""Joins from api_cache/ onto the model's frames. Reads the cache only; src/api.py is what calls the API.

Two kinds of data, with different leak rules:
- Adverts and market series are dated. A row counts only if dated on or before the frame's own cutoff or origin.
- Static geography has no date. It is one value per coordinate, so fold rows and final rows get it the same way.
"""
import json
from functools import lru_cache

import numpy as np
import pandas as pd

from . import api
from . import config as C
from . import features
from .load import RENT_MAX, RENT_MIN, load_input, load_leases

COORD_DP = 5  # about one metre


def _bodies(name: str):
    for f in sorted((api.CACHE_DIR / name).glob("*.json")):
        b = json.loads(f.read_text(encoding="utf-8"))
        if not b.get("denied"):
            yield b


@lru_cache(maxsize=1)
def api_adverts() -> pd.DataFrame:
    """Every cached advert, with the date bound checked again on the way out of the cache."""
    rows = [r for b in _bodies("rent_history") for r in b["data"]["results"]]
    a = pd.DataFrame(rows).drop_duplicates(["gnaf_pid", "listing_date", "weekly_rent"])
    a["lease_date"] = pd.to_datetime(a.listing_date)
    assert a.lease_date.notna().all() and a.lease_date.max() <= C.AS_OF, "an advert later than the as-of date is in the cache"
    return a.rename(columns={"gnaf_pid": "gnaf_id", "weekly_rent": "rent"})


def _coord(lat, lng) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays([pd.Series(lat).round(COORD_DP).to_numpy(), pd.Series(lng).round(COORD_DP).to_numpy()])


def extra_adverts(scope: str) -> pd.DataFrame:
    """API adverts the supplied file lacks, shaped like load_leases.

    scope "input": adverts for input flats. scope "building": also Unit adverts at an input building's coordinates.
    A flat and date already in the file is skipped, so the file's rent always wins.
    """
    leases, flats = load_leases(), load_input()
    a = api_adverts()
    a = a[a.rent.between(RENT_MIN, RENT_MAX)]
    have = pd.MultiIndex.from_arrays([leases.gnaf_id.astype(object), leases.lease_date])
    a = a[~pd.MultiIndex.from_arrays([a.gnaf_id.astype(object), a.lease_date]).isin(have)]

    by_flat = flats.set_index(flats.gnaf_id.astype(object))
    inp = a.gnaf_id.isin(by_flat.index)
    place = flats.assign(c=list(_coord(flats.latitude, flats.longitude))).drop_duplicates("c").set_index("c")
    c = pd.Series(list(_coord(a.lat, a.lng)), index=a.index)
    a = a.assign(building_key=a.gnaf_id.map(by_flat.building_key).fillna(c.map(place.building_key)),
                 suburb=a.gnaf_id.map(by_flat.suburb).fillna(c.map(place.suburb)),
                 input_car_space=a.gnaf_id.map(by_flat.car_space).astype(float))
    if scope == "input":
        a = a[inp]
    elif scope == "building":
        a = a[inp | ((a.property_type == "Unit") & a.building_key.notna())]
    else:
        raise ValueError(scope)
    flag = (pd.to_numeric(a.car_spaces, errors="coerce") > 0).astype(float).where(a.car_spaces.notna())
    a = a.assign(gnaf_id=a.gnaf_id.astype("string"), building_key=a.building_key.astype("string"),
                 advert_car_space=flag, car_space_resolved=flag.fillna(a.input_car_space),
                 bedrooms=pd.to_numeric(a.bedrooms, errors="coerce").fillna(a.gnaf_id.map(by_flat.bedrooms)),
                 bathrooms=pd.to_numeric(a.bathrooms, errors="coerce").fillna(a.gnaf_id.map(by_flat.bathrooms)))
    return a[["gnaf_id", "building_key", "suburb", "lease_date", "rent", "bedrooms", "bathrooms",
              "advert_car_space", "input_car_space", "car_space_resolved"]]


@lru_cache(maxsize=4)
def merged_leases(scope: str) -> pd.DataFrame:
    return pd.concat([load_leases(), extra_adverts(scope)], ignore_index=True).sort_values("lease_date", kind="stable").reset_index(drop=True)


def use_history(scope: str | None) -> None:
    """Point feature building at file + API adverts. Only src.features changes, so targets and labels stay the file's."""
    features.load_leases = load_leases if scope is None else (lambda: merged_leases(scope))
