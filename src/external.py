"""Public open data joined to the flats locally. See notes/public_data.md for sources, licences and results.

Downloads sit in DATA_DIR/external/<source>/ and derived tables in DATA_DIR/external/derived/; both are gitignored.
Nothing from the supplied files is ever sent out: sources are fetched in bulk or by bounding box, and joined here.

Builders (build_*) need shapely and pyproj only for geometry, and run once, through local/build_external.py.
attach() needs only pandas, adds the columns to any frame with gnaf_id, building_key and mb, and masks anything
dated after the row's cutoff.
"""
import json
import re

import numpy as np
import pandas as pd

from . import config as C
from .load import load_input, norm_addr

EXT = C.DATA_DIR / "external"
DERIVED = EXT / "derived"
GNAF_VINTAGE = "may2026"  # last release before 31 July 2026; the August 2026 release is too late

GROUPS = {
    "gnaf_floor": ["g_floor", "g_floor_source", "floor_filled"],
    "gnaf_strata": ["g_sp_number", "g_lot", "g_plan_lots", "g_lot_rank", "g_is_strata"],
    "gnaf_site": ["g_site_addresses", "g_flat_num"],
    "gnaf_created": ["g_created_year"],
    "osm": ["osm_levels", "osm_height", "osm_footprint_m2", "osm_gfa_per_flat", "osm_is_residential"],
    "cadastre": ["cad_site_m2", "cad_site_m2_per_flat"],
    "census": ["c_median_rent", "c_median_hh_income", "c_median_age", "c_hh_size", "c_flat_share", "c_rented_share",
               "c_high_rise_share", "mb_dwellings_2021", "mb_people_2021"],
    "poi": ["d_station_m", "d_light_rail_m", "d_ferry_m", "d_supermarket_m", "d_park_m", "d_motorway_m", "d_rail_line_m",
            "n_bus_stops_400m"],
}


# ---- G-NAF -------------------------------------------------------------------------------------------------------

def _gnaf_raw(vintage: str = GNAF_VINTAGE) -> pd.DataFrame:
    g = pd.read_parquet(DERIVED / f"gnaf_{vintage}.parquet")
    g = g[g.DATE_RETIRED.isna()].copy()
    for c in ("FLAT_NUMBER", "LEVEL_NUMBER", "NUMBER_FIRST"):
        g[c] = pd.to_numeric(g[c], errors="coerce")
    g["site"] = g.STREET_LOCALITY_PID.astype("string") + "|" + g.NUMBER_FIRST.astype("string")
    parcel = g.LEGAL_PARCEL_ID.astype("string").str.extract(r"^(?P<lot>[^/]*)/(?:(?P<sec>[^/]*)/)?(?P<plan>[A-Z]+)(?P<num>\d+)$")
    g["plan_type"], g["plan_num"] = parcel.plan, pd.to_numeric(parcel.num, errors="coerce")
    g["lot"] = pd.to_numeric(parcel.lot, errors="coerce")
    return g


def _level_from_gnaf(g: pd.DataFrame) -> pd.Series:
    """Floor from G-NAF's level fields: 0 for ground codes, the number for level and floor codes."""
    t = g.LEVEL_TYPE_CODE.astype("string")
    out = g.LEVEL_NUMBER.where(t.isin(["L", "FL", "LEVEL", "FLOOR"]))
    return out.where(~t.isin(["G", "GROUND", "UG", "LG", "B", "BASEMENT"]), 0.0).astype(float)


def unit_number_floor(flat_num: pd.Series, site: pd.Series) -> pd.Series:
    """Floor read from a unit number such as 1203 or 405, only in sites that clearly number units this way.

    A site qualifies when most of its units have three or four digits, and the last two digits stay small.
    """
    d = pd.DataFrame({"n": flat_num, "s": site})
    d["big"] = d.n >= 100
    d["tail"] = (d.n % 100).where(d.big)
    d["head"] = (d.n // 100).where(d.big)
    s = d.groupby("s").agg(share=("big", "mean"), tail90=("tail", lambda x: x.quantile(0.9)), floors=("head", "nunique"), n=("n", "count"))
    ok = s[(s.share >= 0.7) & (s.tail90 <= 30) & (s.floors >= 2) & (s.n >= 6)].index
    return d["head"].where(d.s.isin(ok) & d.big).astype(float)


def build_gnaf(vintage: str = GNAF_VINTAGE) -> pd.DataFrame:
    g = _gnaf_raw(vintage)
    strata = g.plan_type.eq("SP")
    plan = g[strata].groupby("plan_num").lot.agg(g_plan_lots="nunique", plan_max_lot="max")
    g = g.merge(plan, left_on="plan_num", right_index=True, how="left")
    g["g_site_addresses"] = g.groupby("site").ADDRESS_DETAIL_PID.transform("size")

    level, unit = _level_from_gnaf(g), unit_number_floor(g.FLAT_NUMBER, g.site)
    g["g_floor"] = level.fillna(unit)
    g["g_floor_source"] = np.where(level.notna(), 1.0, np.where(unit.notna(), 2.0, np.nan))
    g["g_sp_number"] = g.plan_num.where(strata)
    g["g_lot"] = g.lot.where(strata)
    g["g_lot_rank"] = (g.lot / g.plan_max_lot).where(strata)
    g["g_is_strata"] = strata.astype(float).where(g.LEGAL_PARCEL_ID.notna())
    g["g_flat_num"] = g.FLAT_NUMBER
    g["g_created"] = pd.to_datetime(g.DATE_CREATED, errors="coerce")
    g["g_mb"] = g.MB_2021_CODE.astype("string")
    out = g.rename(columns={"ADDRESS_DETAIL_PID": "gnaf_id"})[
        ["gnaf_id", "g_floor", "g_floor_source", "g_sp_number", "g_lot", "g_plan_lots", "g_lot_rank", "g_is_strata",
         "g_site_addresses", "g_flat_num", "g_created", "g_mb"]]
    out = out[out.gnaf_id.isin(set(load_input().gnaf_id.astype(str)))].drop_duplicates("gnaf_id")
    out.to_parquet(DERIVED / f"t_gnaf_{vintage}.parquet")
    return out


# ---- OSM buildings -----------------------------------------------------------------------------------------------

def _num(v) -> float:
    m = re.match(r"\s*(-?\d+(?:\.\d+)?)", str(v)) if v is not None else None
    return float(m.group(1)) if m else float("nan")


def build_osm() -> pd.DataFrame:
    """Each building_key takes the footprint under most of its flats. Heights fall back to parts inside the footprint."""
    import shapely
    from pyproj import Transformer

    tf = Transformer.from_crs(4326, 7856, always_xy=True)  # GDA2020 MGA zone 56, metres
    els = json.load(open(EXT / "osm_overpass_20260731" / "buildings.json", encoding="utf-8"))["elements"]

    def ring(pts):
        x, y = tf.transform([p["lon"] for p in pts], [p["lat"] for p in pts])
        return shapely.Polygon(list(zip(x, y))) if len(x) >= 4 else None

    rows = []
    for e in els:
        t = e.get("tags", {})
        if e["type"] == "way":
            polys = [ring(e.get("geometry", []))]
        else:
            polys = [ring(m.get("geometry", [])) for m in e.get("members", []) if m.get("role") == "outer" and m.get("type") == "way"]
        polys = [p if p.is_valid else p.buffer(0) for p in polys if p is not None]
        if not polys:
            continue
        rows.append(dict(geom=shapely.union_all(polys), kind=t.get("building"), part="building:part" in t,
                         levels=_num(t.get("building:levels")), height=_num(t.get("height"))))
    b = pd.DataFrame(rows)
    b = b[~shapely.is_empty(b.geom.to_numpy())].reset_index(drop=True)
    parts = b[b.part | b.kind.isna()].reset_index(drop=True)
    foot = b[b.kind.notna() & ~b.kind.isin(["roof", "garage", "garages", "shed", "carport"])].reset_index(drop=True)

    tree = shapely.STRtree(parts.geom.to_numpy())
    pi, fi = tree.query(foot.geom.to_numpy(), predicate="intersects")
    pl = pd.DataFrame({"f": pi, "levels": parts.levels.to_numpy()[fi], "height": parts.height.to_numpy()[fi]}).groupby("f").max()
    foot["levels"] = foot.levels.combine(pl.levels.reindex(foot.index), lambda a, c: np.nanmax([a, c]) if not (np.isnan(a) and np.isnan(c)) else np.nan)
    foot["height"] = foot.height.combine(pl.height.reindex(foot.index), lambda a, c: np.nanmax([a, c]) if not (np.isnan(a) and np.isnan(c)) else np.nan)
    foot["area"] = shapely.area(foot.geom.to_numpy())

    f = load_input()[["gnaf_id", "building_key", "latitude", "longitude", "flats_in_building"]].dropna(subset=["latitude", "longitude"])
    x, y = tf.transform(f.longitude.to_numpy(), f.latitude.to_numpy())
    pts = shapely.points(x, y)
    tree = shapely.STRtree(foot.geom.to_numpy())
    idx, dist = tree.query_nearest(pts, max_distance=15, return_distance=True, all_matches=False)
    m = pd.DataFrame({"row": idx[0], "foot": idx[1], "dist": dist}).drop_duplicates("row")
    f = f.iloc[m.row.to_numpy()].assign(foot=m.foot.to_numpy(), dist=m.dist.to_numpy())
    best = f.groupby("building_key").foot.agg(lambda s: s.mode().iat[0])
    out = pd.DataFrame({"building_key": best.index.astype(str), "foot": best.to_numpy()})
    src = foot.iloc[out.foot.to_numpy()].reset_index(drop=True)
    out["osm_levels"], out["osm_height"], out["osm_footprint_m2"] = src.levels, src.height, src.area
    out["osm_is_residential"] = src.kind.isin(["apartments", "residential", "terrace", "house", "detached"]).astype(float)
    flats = load_input().groupby("building_key").flats_in_building.max().reindex(out.building_key).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out["osm_gfa_per_flat"] = np.where(flats > 0, out.osm_footprint_m2 * out.osm_levels / flats, np.nan)
    out = out.drop(columns="foot")
    out.to_parquet(DERIVED / "t_osm.parquet")
    return out


# ---- NSW cadastre ------------------------------------------------------------------------------------------------

def build_cadastre() -> pd.DataFrame:
    """Site area of each strata plan from the Lot layer, by plan label. Only lots that existed by AS_OF are kept."""
    feats = json.load(open(EXT / "nsw_cadastre" / "lots_bbox.geojson", encoding="utf-8"))["features"]
    p = pd.DataFrame([f["properties"] for f in feats])
    created = pd.to_datetime(p.createdate, unit="ms", errors="coerce")
    p = p[created.isna() | (created <= C.AS_OF)]
    p = p[p.planlabel.astype("string").str.startswith("SP", na=False)].copy()
    p["plan_num"] = pd.to_numeric(p.planlabel.str.extract(r"(\d+)")[0], errors="coerce")
    out = p.groupby("plan_num").planlotarea.max().rename("cad_site_m2").reset_index()
    out = out[out.cad_site_m2 > 0]
    out.to_parquet(DERIVED / "t_cadastre.parquet")
    return out


# ---- OSM transport and amenities ---------------------------------------------------------------------------------

def build_poi() -> pd.DataFrame:
    import shapely
    from pyproj import Transformer

    tf = Transformer.from_crs(4326, 7856, always_xy=True)
    els = json.load(open(EXT / "osm_overpass_poi" / "poi.json", encoding="utf-8"))["elements"]
    rows = []
    for e in els:
        c = e.get("center") or e
        if "lat" in c:
            rows.append(dict(lat=c["lat"], lon=c["lon"], **{k: e.get("tags", {}).get(k) for k in ("railway", "station", "light_rail", "amenity", "highway", "leisure", "shop")}))
    p = pd.DataFrame(rows)
    p["x"], p["y"] = tf.transform(p.lon.to_numpy(), p.lat.to_numpy())
    f = load_input()[["gnaf_id", "latitude", "longitude"]].dropna().reset_index(drop=True)
    fx, fy = tf.transform(f.longitude.to_numpy(), f.latitude.to_numpy())
    pts = shapely.points(fx, fy)
    # Sydney light rail stops are tagged railway=station with station=light_rail, not tram_stop.
    light = p.station.eq("light_rail") | p.light_rail.eq("yes") | p.railway.eq("tram_stop")
    sets = dict(d_station_m=p.railway.isin(["station", "halt"]) & ~light, d_light_rail_m=light, d_ferry_m=p.amenity.eq("ferry_terminal"),
                d_supermarket_m=p.shop.eq("supermarket"), d_park_m=p.leisure.eq("park"), d_motorway_m=p.highway.isin(["motorway", "trunk"]),
                d_rail_line_m=p.railway.eq("rail"))
    out = f[["gnaf_id"]].copy()
    for name, mask in sets.items():
        q = p[mask]
        tree = shapely.STRtree(shapely.points(q.x.to_numpy(), q.y.to_numpy()))
        _, dist = tree.query_nearest(pts, return_distance=True, all_matches=False)
        idx = tree.query_nearest(pts, all_matches=False)[0]
        out[name] = pd.Series(dist).groupby(idx).first().reindex(range(len(f))).to_numpy()
    bus = p[p.highway.eq("bus_stop")]
    tree = shapely.STRtree(shapely.points(bus.x.to_numpy(), bus.y.to_numpy()))
    row, _ = tree.query(pts, predicate="dwithin", distance=400)
    out["n_bus_stops_400m"] = np.bincount(row, minlength=len(f))
    out.to_parquet(DERIVED / "t_poi.parquet")
    return out


# ---- Census ------------------------------------------------------------------------------------------------------

def build_census() -> pd.DataFrame:
    """2021 Census by SA1 and 2021 mesh block counts, keyed by mesh block."""
    import zipfile

    d = EXT / "abs_census2021"
    alloc = pd.read_excel(d / "MB_2021_AUST.xlsx", usecols=["MB_CODE_2021", "SA1_CODE_2021", "STATE_CODE_2021"], dtype=str)
    alloc = alloc[alloc.STATE_CODE_2021 == "1"].drop(columns="STATE_CODE_2021")

    parts = []
    for sheet in ("Table 1", "Table 1.1"):  # New South Wales, parts 1 and 2
        raw = pd.read_excel(d / "mb_counts_2021.xlsx", sheet_name=sheet, header=None, dtype=str)
        head = raw.index[raw.iloc[:, 0].astype("string").str.contains("MB_CODE", na=False)][0]
        t = raw.iloc[head + 1:].set_axis(raw.iloc[head].astype(str).str.strip().str.upper(), axis=1)
        parts.append(pd.DataFrame({"MB_CODE_2021": t.filter(like="MB_CODE").iloc[:, 0],
                                   "mb_dwellings_2021": pd.to_numeric(t.filter(like="DWELLING").iloc[:, 0], errors="coerce"),
                                   "mb_people_2021": pd.to_numeric(t.filter(like="PERSON").iloc[:, 0], errors="coerce")}))
    mbc = pd.concat(parts).dropna(subset=["MB_CODE_2021"])

    z = zipfile.ZipFile(d / "gcp_sa1_nsw.zip")

    def table(t):
        n = next(n for n in z.namelist() if re.search(rf"_{t}_NSW_SA1\.csv$", n))
        with z.open(n) as f:
            return pd.read_csv(f, dtype={"SA1_CODE_2021": str}).set_index("SA1_CODE_2021")

    g2, dw, ten = table("G02"), table("G36"), table("G37")
    s = pd.DataFrame({"c_median_rent": g2.Median_rent_weekly, "c_median_hh_income": g2.Median_tot_hhd_inc_weekly,
                      "c_median_age": g2.Median_age_persons, "c_hh_size": g2.Average_household_size})
    s = s.where(s > 0)
    occupied = dw.OPDs_Tot_OPDs_Dwellings.replace(0, np.nan)
    # ABS perturbs small cells, so a share can pass 1.
    s["c_flat_share"] = (dw.OPDs_Flt_apart_Tot_Dwgs / occupied).clip(upper=1)
    s["c_high_rise_share"] = (dw.OPDs_F_ap_I_9_m_sty_blk_Ds / occupied).clip(upper=1)
    s["c_rented_share"] = (ten.R_Tot_Total / ten.filter(regex="^Total").iloc[:, -1].replace(0, np.nan)).clip(upper=1)
    out = alloc.merge(s, left_on="SA1_CODE_2021", right_index=True, how="left").merge(mbc, on="MB_CODE_2021", how="left")
    out = out.rename(columns={"MB_CODE_2021": "mb_key"}).drop(columns="SA1_CODE_2021")
    out.to_parquet(DERIVED / "t_census.parquet")
    return out


# ---- Join --------------------------------------------------------------------------------------------------------

def _read(name):
    p = DERIVED / name
    return pd.read_parquet(p) if p.exists() else None


def attach(df: pd.DataFrame, cutoff=None, vintage: str = GNAF_VINTAGE) -> pd.DataFrame:
    """Adds every built table. cutoff is one date, or None to use each row's `origin`. Row order and count are kept."""
    n = len(df)
    out = df.copy()
    when = pd.to_datetime(out["origin"]) if cutoff is None else pd.Series(pd.Timestamp(cutoff), index=out.index)
    ids = out.gnaf_id.astype("object")

    g = _read(f"t_gnaf_{vintage}.parquet")
    if g is not None:
        g = g.assign(gnaf_id=g.gnaf_id.astype("object")).set_index("gnaf_id")
        for c in g.columns:
            out[c] = ids.map(g[c]).to_numpy()
        created = pd.to_datetime(out.pop("g_created"))
        # Dated, so masked: a row must not learn that an address appears after its cutoff.
        out["g_created_year"] = created.dt.year.where(created <= when).astype(float)
        floor = pd.to_numeric(out.get("floor_level"), errors="coerce")
        out["floor_filled"] = floor.fillna(out.g_floor) if "floor_level" in out else out.g_floor
        mb_key = out.g_mb.astype("object").fillna(out["mb"].astype("object"))
    else:
        mb_key = out["mb"].astype("object")

    c = _read("t_cadastre.parquet")
    if c is not None and "g_sp_number" in out:
        out["cad_site_m2"] = out.g_sp_number.map(c.set_index("plan_num").cad_site_m2).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            out["cad_site_m2_per_flat"] = out.cad_site_m2 / out.g_plan_lots.where(out.g_plan_lots > 0)

    o = _read("t_osm.parquet")
    if o is not None:
        o = o.assign(building_key=o.building_key.astype("object")).set_index("building_key")
        bk = out.building_key.astype("object")
        for col in o.columns:
            out[col] = bk.map(o[col]).to_numpy()

    p = _read("t_poi.parquet")
    if p is not None:
        p = p.assign(gnaf_id=p.gnaf_id.astype("object")).set_index("gnaf_id")
        for col in p.columns:
            out[col] = ids.map(p[col]).to_numpy()

    s = _read("t_census.parquet")
    if s is not None:
        s = s.assign(mb_key=s.mb_key.astype("object")).drop_duplicates("mb_key").set_index("mb_key")
        for col in s.columns:
            out[col] = mb_key.map(s[col]).to_numpy()

    assert len(out) == n
    return out


def blank_rates(fold_rows: pd.DataFrame, final_rows: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """The skew test: a feature blank far more often in fold rows than in final rows cannot be judged by the folds."""
    t = pd.DataFrame({"fold_blank": fold_rows[cols].isna().mean(), "final_blank": final_rows[cols].isna().mean()})
    t["gap"] = (t.fold_blank - t.final_blank).abs()
    return t.round(4)


def norm_building(s: pd.Series) -> pd.Series:
    return norm_addr(s)
