"""Joins from api_cache/ onto the model's frames. Reads the cache only; src/api.py is what calls the API.

Two kinds of data, with different leak rules:
- Adverts, sales and market series are dated. A row counts only if dated on or before the frame row's own origin.
- Static geography has no date. It is one value per building, so fold rows and final rows get it the same way.
"""
import json
import re
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import api
from . import config as C
from . import features
from .load import RENT_MAX, RENT_MIN, load_input, load_leases

COORD_DP = 5  # about one metre


# ---- Adverts ----

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


# ---- Static geography: one row per building, from one representative flat ----

# Named light rail stops, plus the unnamed "Train station" points, which all lie along the Inner West and George St lines.
LIGHT_RAIL = re.compile(r"light rail|moore park|surry hills|glebe railway tunnel|bridge street|chalmers|chinatown|haymarket|qvb|kingsford|^train station$", re.I)
ROAD_GROUPS = dict(road_major_m=("motorway", "trunk", "primary"), road_any_m=None)
GEO_COLS = ["train_m", "light_rail_m", "bus_m", "ferry_m", "bus_stops_2km", "supermarket_m", "green_m", "restaurant_m",
            "supermarkets_1km", "green_1km", "restaurants_1km", "restaurants_300m", "school_m", "public_primary_m",
            "public_secondary_m", "schools_1km", "private_schools_1km", "heritage_on", "heritage_m", "flood_on", "flood_m",
            "hv_line_m", "environment_m", "public_housing_share", "road_major_m", "road_any_m", "noise", "noise_max_150m",
            "height_limit_m", "fsr_limit", "zoning"]


def _xy(lat, lng) -> np.ndarray:
    """Metres on a flat local grid. Good to well under a metre across the City of Sydney."""
    return np.column_stack([(np.asarray(lng, float) - 151.2) * 111_320 * np.cos(np.radians(33.88)),
                            (np.asarray(lat, float) + 33.88) * 110_574])


def _dist_to_points(xy: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return cKDTree(pts).query(xy)[0] if len(pts) else np.full(len(xy), np.nan)


def _dist_to_segments(xy: np.ndarray, a: np.ndarray, b: np.ndarray, chunk: int = 128) -> np.ndarray:
    """Least distance from each point to any segment a-b."""
    out = np.full(len(xy), np.nan)
    if not len(a):
        return out
    ab = b - a
    n2 = np.maximum((ab * ab).sum(1), 1e-9)
    for i in range(0, len(xy), chunk):
        p = xy[i:i + chunk, None, :]
        t = np.clip(((p - a) * ab).sum(2) / n2, 0, 1)
        out[i:i + chunk] = np.sqrt((((a + t[..., None] * ab) - p) ** 2).sum(2)).min(1)
    return out


def _rings(geom: dict) -> list:
    if geom["type"] == "Polygon":
        return [np.asarray(r, float) for r in geom["coordinates"]]
    if geom["type"] == "MultiPolygon":
        return [np.asarray(r, float) for p in geom["coordinates"] for r in p]
    return []


def _inside(lng: np.ndarray, lat: np.ndarray, rings: list) -> np.ndarray:
    """Even-odd ray test over all of a shape's rings, so holes come out."""
    odd = np.zeros(len(lng), bool)
    for r in rings:
        x0, y0, x1, y1 = r[:-1, 0], r[:-1, 1], r[1:, 0], r[1:, 1]
        near = (lng >= r[:, 0].min()) & (lng <= r[:, 0].max()) & (lat >= r[:, 1].min()) & (lat <= r[:, 1].max())
        for i in np.flatnonzero(near):
            crosses = (y0 > lat[i]) != (y1 > lat[i])
            with np.errstate(divide="ignore", invalid="ignore"):
                odd[i] ^= np.count_nonzero(crosses & (lng[i] < (x1 - x0) * (lat[i] - y0) / (y1 - y0) + x0)) % 2 == 1
    return odd


def _at_points(lng, lat, feats: list, field: str) -> np.ndarray:
    """A polygon property at each point; where zones overlap at a suburb edge, the highest value."""
    out = np.full(len(lng), np.nan)
    for f in feats:
        v = f["properties"].get(field)
        if v is not None:
            hit = _inside(lng, lat, _rings(f["geometry"]))
            out[hit] = np.fmax(out[hit], float(v))
    return out


def _nearest(items: list, keep) -> float:
    d = [x["distance_m"] for x in items if keep(x) and x.get("distance_m") is not None]
    return float(min(d)) if d else np.nan


def _risk_m(block) -> float:
    v = [x["nearest_distance_m"] for x in block or [] if x.get("nearest_distance_m") is not None]
    return float(min(v)) if v else np.nan


def _on(block) -> float:
    return float(any(x.get("on_property") for x in block or []))


def _share(block) -> float:
    """Public housing share of the nearest area."""
    if not block:
        return np.nan
    m = re.match(r"([\d.]+)%", str(min(block, key=lambda r: r.get("nearest_distance_m") or 0).get("value")))
    return float(m.group(1)) if m else np.nan


def _building_row(gnaf: str) -> dict:
    """Everything that comes from per-property calls. Absent responses leave the fields blank."""
    p = f"properties/{gnaf}"
    r = {}
    t = api.cached(f"{p}/transport/nearby", dict(radius_m=2000))
    if t:
        n = t["data"]["nearest"]
        r.update(bus_m=(n.get("bus") or {}).get("distance_m", np.nan), ferry_m=(n.get("ferry") or {}).get("distance_m", np.nan),
                 bus_stops_2km=t["data"]["counts"].get("bus", np.nan))
    a = api.cached(f"{p}/amenities/nearby")
    if a:
        c, res = a["data"]["counts"], a["data"]["results"]
        r.update(supermarkets_1km=c.get("supermarket", 0), green_1km=c.get("green_spaces", 0), restaurants_1km=c.get("restaurant", 0),
                 supermarket_m=_nearest(res, lambda x: x["category"] == "Supermarket"),
                 green_m=_nearest(res, lambda x: x["category"] == "Green Spaces"),
                 restaurant_m=_nearest(res, lambda x: x["category"] == "Restaurant"),
                 restaurants_300m=sum(x["category"] == "Restaurant" and x["distance_m"] <= 300 for x in res))
    s = api.cached(f"{p}/schools/nearby")
    if s:
        d = s["data"]
        public = lambda x: x.get("sector") == "public"
        r.update(school_m=_nearest(d, lambda x: True),
                 public_primary_m=_nearest(d, lambda x: public(x) and x.get("school_level_type") in ("Primary", "Combined")),
                 public_secondary_m=_nearest(d, lambda x: public(x) and x.get("school_level_type") in ("Secondary", "Combined")),
                 schools_1km=sum(x["distance_m"] <= 1000 for x in d),
                 private_schools_1km=sum(x["distance_m"] <= 1000 and not public(x) for x in d))
    k = api.cached(f"{p}/risks/all")
    if k:
        d = k["data"]
        r.update(heritage_on=_on(d.get("heritage")), heritage_m=_risk_m(d.get("heritage")), flood_on=_on(d.get("flood")),
                 flood_m=_risk_m(d.get("flood")), hv_line_m=_risk_m(d.get("hv_power_line")), environment_m=_risk_m(d.get("environment")),
                 public_housing_share=_share(d.get("public_housing")))
    z = api.cached(f"{p}/development/all")
    if z and z["data"].get("zoning_code"):
        r["zoning"] = z["data"]["zoning_code"].get("name")
    return r


def _covered() -> list:
    return api.cached("me")["area"]["suburbs"]


def _reps() -> pd.DataFrame:
    f = load_input()
    return f[f.suburb.isin(_covered())].drop_duplicates(["suburb", "building_key"])


def stations() -> pd.DataFrame:
    """Rail stops seen as any building's nearest. `nearest.train` mixes heavy and light rail and `tram` is always null,
    so the two are split here by name and each building's distance is measured to both."""
    rows = {}
    for g in _reps().gnaf_id:
        t = api.cached(f"properties/{g}/transport/nearby", dict(radius_m=2000))
        x = t and t["data"]["nearest"].get("train")
        if x:
            rows[(x["lat"], x["lng"])] = x.get("name") or x.get("label")
    s = pd.DataFrame([(k[0], k[1], v) for k, v in rows.items()], columns=["lat", "lng", "name"])
    return s.assign(light=s.name.str.contains(LIGHT_RAIL))


def _suburb_data(tail: str) -> list:
    return [b["data"] for b in (api.cached(api.suburb_path(s, tail)) for s in _covered()) if b and b.get("data")]


@lru_cache(maxsize=1)
def geo_table() -> pd.DataFrame:
    """One row per (suburb, building_key), for the covered suburbs."""
    reps = _reps()
    t = pd.DataFrame([_building_row(str(g)) for g in reps.gnaf_id], index=reps.index)
    xy, lat, lng = _xy(reps.latitude, reps.longitude), reps.latitude.to_numpy(), reps.longitude.to_numpy()

    s = stations()
    t["train_m"] = _dist_to_points(xy, _xy(s[~s.light].lat, s[~s.light].lng))
    t["light_rail_m"] = _dist_to_points(xy, _xy(s[s.light].lat, s[s.light].lng))

    lines = [(f["properties"].get("road_class"), np.asarray(c, float)) for d in _suburb_data("shapes/busy-roads") for f in d["geojson"]["features"]
             for c in ([f["geometry"]["coordinates"]] if f["geometry"]["type"] == "LineString" else f["geometry"]["coordinates"])]
    for col, classes in ROAD_GROUPS.items():
        pts = [_xy(c[:, 1], c[:, 0]) for cl, c in lines if classes is None or cl in classes]
        t[col] = _dist_to_segments(xy, np.concatenate([p[:-1] for p in pts]), np.concatenate([p[1:] for p in pts]))

    heat = np.array([h for d in _suburb_data("shapes/noise-heatmap") for h in d.get("heat", [])], float)
    tree = cKDTree(_xy(heat[:, 0], heat[:, 1]))
    dist, idx = tree.query(xy)
    t["noise"] = np.where(dist <= 100, heat[idx, 2], np.nan)
    t["noise_max_150m"] = [heat[i, 2].max() if len(i) else np.nan for i in tree.query_ball_point(xy, 150)]

    controls = _suburb_data("development/building-controls")
    t["height_limit_m"] = _at_points(lng, lat, [f for d in controls for f in d["hob"]["geojson"]["features"]], "height_m")
    t["fsr_limit"] = _at_points(lng, lat, [f for d in controls for f in d["fsr"]["geojson"]["features"]], "fsr")
    t = t.reindex(columns=GEO_COLS)
    return t.assign(suburb=reps.suburb.astype(object), building_key=reps.building_key.astype(object))


@lru_cache(maxsize=1)
def crime_table() -> pd.Series:
    """Modelled crime rate by mesh block. See notes/api_features.md: its training period is stated, its inputs are not."""
    return pd.Series({mb: v for d in _suburb_data("crime/by-mesh-block") for mb, v in d["mesh_blocks"].items()}, dtype=float)


def add_geo(frame: pd.DataFrame) -> pd.DataFrame:
    """Add GEO_COLS and crime_rate. Rows keep their order; buildings outside the covered suburbs stay blank."""
    keys = pd.DataFrame({"suburb": frame.suburb.astype(object).to_numpy(), "building_key": frame.building_key.astype(object).to_numpy()})
    got = keys.merge(geo_table(), on=["suburb", "building_key"], how="left", validate="m:1").drop(columns=["suburb", "building_key"])
    got.index = frame.index
    return pd.concat([frame, got], axis=1).assign(crime_rate=frame.mb.astype(object).map(crime_table()).to_numpy())


# ---- Dated features: only rows dated on or before each frame row's own origin ----

SALE_COLS = ["sale_log_price", "sale_age_days", "sale_n", "sale_rel_building", "bldg_sale_log_price", "bldg_sale_n"]
MARKET_COLS = ["mkt_rent_chg_3m", "mkt_rent_chg_12m", "mkt_vacancy", "mkt_vacancy_chg_6m"]


@lru_cache(maxsize=1)
def sales_table() -> pd.DataFrame:
    """Sold records with a price, one row per flat and date, checked against the as-of date."""
    rows = []
    for g in load_input().gnaf_id.astype(str):
        b = api.cached(f"properties/{g}/sale-history")
        for x in (b or {}).get("data") or []:
            if x.get("type") == "Sold" and x.get("price"):
                rows.append((g, x["date"], float(x["price"])))
    s = pd.DataFrame(rows, columns=["gnaf_id", "date", "price"]).drop_duplicates(["gnaf_id", "date"])
    s["date"] = pd.to_datetime(s.date)
    assert s.date.max() <= C.AS_OF, "a sale later than the as-of date is in the cache"
    return s[s.price >= 50_000].sort_values("date").reset_index(drop=True)


def add_sales(frame: pd.DataFrame, origin_col: str = "origin") -> pd.DataFrame:
    """A flat's last sale and its building's sales in the prior five years, as of each row's origin.

    sale_rel_building compares the two: a price-based stand-in for size and quality that the adverts do not carry.
    """
    flats = load_input()
    to_bldg = dict(zip(flats.gnaf_id.astype(object), flats.building_key.astype(object)))
    s = sales_table().assign(lp=lambda t: np.log(t.price))
    s["building_key"] = s.gnaf_id.map(to_bldg)
    out = pd.DataFrame(index=frame.index, columns=SALE_COLS, dtype=float)
    for origin, rows in frame.groupby(origin_col, sort=False):
        o = pd.Timestamp(origin)
        h = s[s.date <= o]
        assert h.date.max() <= o
        last = h.groupby("gnaf_id").agg(lp=("lp", "last"), date=("date", "last"), n=("lp", "size"))
        g = rows.gnaf_id.astype(object)
        out.loc[rows.index, "sale_log_price"] = g.map(last.lp).to_numpy()
        out.loc[rows.index, "sale_age_days"] = (o - g.map(last.date)).dt.days.to_numpy()
        out.loc[rows.index, "sale_n"] = g.map(last.n).fillna(0).to_numpy()
        b = h[h.date > o - pd.DateOffset(years=5)].groupby("building_key").lp.agg(["median", "size"])
        k = rows.building_key.astype(object)
        out.loc[rows.index, "bldg_sale_log_price"] = k.map(b["median"]).to_numpy()
        out.loc[rows.index, "bldg_sale_n"] = k.map(b["size"]).fillna(0).to_numpy()
    out["sale_rel_building"] = out.sale_log_price - out.bldg_sale_log_price
    return pd.concat([frame, out], axis=1)


def series(tail: str, field: str = "unit") -> pd.DataFrame:
    rows = [(d["area_name"], x["date"], x["value"]) for d in _suburb_data(tail) for x in d.get(field) or []]
    t = pd.DataFrame(rows, columns=["suburb", "date", "value"])
    t["date"] = pd.to_datetime(t.date)
    assert t.date.max() <= C.AS_OF
    return t.dropna().sort_values("date")


def _value_at(t: pd.DataFrame, when: pd.Timestamp) -> pd.Series:
    """Each suburb's latest value dated on or before `when`."""
    return t[t.date <= when].groupby("suburb").value.last()


def add_market(frame: pd.DataFrame, origin_col: str = "origin") -> pd.DataFrame:
    """Suburb unit rent momentum and vacancy, read at each row's origin. Nothing after the origin is reachable."""
    rent, vac = series("market/median-rent-series"), series("market/vacancy-rate-series")
    out = pd.DataFrame(index=frame.index, columns=MARKET_COLS, dtype=float)
    for origin, rows in frame.groupby(origin_col, sort=False):
        o = pd.Timestamp(origin)
        sub = rows.suburb.astype(object)
        now, then3, then12 = (_value_at(rent, o - pd.DateOffset(months=m)) for m in (0, 3, 12))
        v_now, v_then = _value_at(vac, o), _value_at(vac, o - pd.DateOffset(months=6))
        out.loc[rows.index, "mkt_rent_chg_3m"] = sub.map(np.log(now / then3)).to_numpy()
        out.loc[rows.index, "mkt_rent_chg_12m"] = sub.map(np.log(now / then12)).to_numpy()
        out.loc[rows.index, "mkt_vacancy"] = sub.map(v_now).to_numpy()
        out.loc[rows.index, "mkt_vacancy_chg_6m"] = sub.map(v_now - v_then).to_numpy()
    return pd.concat([frame, out], axis=1)
