"""build_features(cutoff_date, target_rows): the one place features are made.

Nothing dated after cutoff_date may reach a feature. That includes input columns that
carry later information: ever_advertised_for_lease, attrs_last_seen, first_ever_listed.

Anchors are in dollars at target_date. Each past rent is deflated to the cutoff with the
index, normalised to "no parking" with a premium estimated inside buildings, then projected
to target_date with the target's own parking added back. So flipping car_space_resolved
moves every anchor, which the two-pass model needs: a flat's parking almost never changes
between its adverts, so a model could not learn that step from anchor flags alone.
Everything per-cutoff lives in _prepare and is cached; build_features only does lookups.
"""
import numpy as np
import pandas as pd

from . import config as C
from .index import DEFAULT_PARAMS, IndexParams, memo, rent_index
from .load import load_blocks, load_input, load_leases

BASELINE_LEVELS = [["suburb", "bedrooms", "car_space_resolved"], ["suburb", "bedrooms"], ["bedrooms"], []]
MIN_GROUP = 5

HALF_LIFE_MONTHS = 12    # building comparables: weight halves every 12 months back from the cutoff
BLOCK_WINDOW_MONTHS = 12
MIN_BLOCK_ROWS = 5
CLIP_LOG = 0.4           # building comparables are limited to this far from their cell median
PREMIUM_PRIOR = 200      # rows of within-building evidence a suburb needs to pull half way off the city premium
MAX_FLOOR_ADJ = 0.3
BED_SLOTS = 32

ANCHORS = ["anchor_own", "anchor_bldg", "anchor_block"]
ANCHOR_COLS = [f"{a}{s}" for a in ANCHORS for s in ("", "_age_days", "_n")]
BLOCK_EXTRA = ["block_people_2026", "block_people_per_home_2026", "block_homes_growth"]
FEATURE_COLS = [
    "bedrooms", "bathrooms", "car_space_resolved", "floor_level", "flats_in_building",
    "block_new_homes_2021_2026", "block_homes_2026", *BLOCK_EXTRA, "latitude", "longitude", "suburb",
    "attrs_age_days", "listed_age_days", "ever_advertised_for_lease",
    "floor_level_missing", "bathrooms_missing", "input_car_space_missing", "block_missing", "attrs_missing", "listed_missing",
    *ANCHOR_COLS, "anchor_own_parking", "anchor_own_parking_match", "anchor_bldg_parking_match", "anchor_block_level",
    "best_anchor", "best_anchor_source",
]
CATEGORICAL_COLS = ["suburb"]


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
    seen = pd.Index(history_upto(cutoff).gnaf_id.dropna().unique(), dtype="object")
    f["ever_advertised_for_lease"] = (_codes(seen, f.gnaf_id) >= 0).astype("int8")  # Series.isin is very slow on Arrow strings
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


def _sums(keys: np.ndarray, cols: np.ndarray):
    """Sum the columns of cols by integer key, ignoring negative keys. Returns sorted keys and sums."""
    ok = keys >= 0
    u, inv = np.unique(keys[ok], return_inverse=True)
    return u, np.stack([np.bincount(inv, weights=c, minlength=len(u)) for c in cols[ok].T], axis=1)


def _find(table, keys: np.ndarray) -> np.ndarray:
    """Rows of a (sorted keys, values) table for each key; zero rows where there is no match."""
    u, vals = table
    out = np.zeros((len(keys), vals.shape[1]))
    if len(u):
        pos = np.minimum(np.searchsorted(u, keys), len(u) - 1)
        hit = (keys >= 0) & (u[pos] == keys)
        out[hit] = vals[pos[hit]]
    return out


def _compound(a: np.ndarray, b: np.ndarray, size: int) -> np.ndarray:
    """Join two non-negative integer codes into one; -1 if either is missing."""
    return np.where((a >= 0) & (b >= 0), a * size + b, -1)


def _codes(index: pd.Index, values) -> np.ndarray:
    return index.get_indexer(pd.Series(values).astype("object").to_numpy())


def _bed_code(bedrooms) -> np.ndarray:
    b = pd.to_numeric(pd.Series(bedrooms), errors="coerce").to_numpy(dtype=float)
    return np.where(np.isnan(b), -1, np.clip(np.nan_to_num(b).round(), 0, BED_SLOTS - 1)).astype("int64")


def _within(values: np.ndarray, cell: np.ndarray) -> np.ndarray:
    return values - pd.Series(values).groupby(cell).transform("mean").to_numpy()


def _parking_premium(a, q, cell, suburb_row, n_rows):
    """Log premium for a car space, by index row, from flats compared inside building x bedrooms.

    Suburbs are shrunk to the city figure, and the last row is the city figure.
    """
    ok = (cell >= 0) & ~np.isnan(q)
    if ok.sum() < 2:
        return np.zeros(n_rows)
    x, y = _within(q[ok], cell[ok]), _within(a[ok], cell[ok])
    y = np.clip(y, -CLIP_LOG, CLIP_LOG)
    sxx = np.bincount(suburb_row[ok], weights=x * x, minlength=n_rows)
    sxy = np.bincount(suburb_row[ok], weights=x * y, minlength=n_rows)
    city = sxy.sum() / sxx.sum() if sxx.sum() > 0 else 0.0
    return np.maximum((sxy + PREMIUM_PRIOR * 0.25 * city) / (sxx + PREMIUM_PRIOR * 0.25), 0.0)


def _floor_slope(a, floor, cell) -> float:
    """Log rent per floor inside building x bedrooms, never negative."""
    ok = (cell >= 0) & ~np.isnan(floor)
    if ok.sum() < 2:
        return 0.0
    x, y = _within(floor[ok], cell[ok]), _within(a[ok], cell[ok])
    sxx = (x * x).sum()
    return max(float((x * np.clip(y, -CLIP_LOG, CLIP_LOG)).sum() / sxx), 0.0) if sxx > 0 else 0.0


def _prepare(leases: pd.DataFrame, cutoff: pd.Timestamp, params: IndexParams) -> dict:
    """Everything that depends on the cutoff and not on the target rows."""
    index = rent_index(leases, cutoff, params)
    h = leases[leases.lease_date <= cutoff]
    flats = load_input()
    gnaf_ix = pd.Index(h.gnaf_id.dropna().unique(), dtype="object")
    bldg_ix = pd.Index(h.building_key.dropna().unique(), dtype="object")

    rows = index.codes(h.suburb)
    a = np.log(h.rent.to_numpy(dtype=float)) - index.at(h.lease_date, h.suburb)  # log rent in cutoff money
    q = h.car_space_resolved.to_numpy(dtype=float)
    q = np.where(np.isnan(q), np.nan, (q > 0).astype(float))
    gnaf, bldg, bed = _codes(gnaf_ix, h.gnaf_id), _codes(bldg_ix, h.building_key), _bed_code(h.bedrooms)
    cell = _compound(bldg, bed, BED_SLOTS)

    premium = _parking_premium(a, q, cell, rows, len(index.groups) + 1)
    share = float(np.nanmean(q)) if (~np.isnan(q)).any() else 0.5  # unknown parking counts as the average
    a0 = a - premium[rows] * np.where(np.isnan(q), share, q)
    floor = pd.to_numeric(h.gnaf_id.map(flats.set_index("gnaf_id").floor_level), errors="coerce").to_numpy(dtype=float)
    floor_slope = _floor_slope(a0, floor, cell)

    age = (cutoff - h.lease_date).dt.days.to_numpy(dtype=float)  # days before the cutoff

    # Own flat: the latest rent; several adverts on that last day count as their median.
    d = pd.DataFrame({"g": gnaf, "age": age, "a": a, "q": q})
    d = d[d.g >= 0]
    last = d[d.age == d.groupby("g").age.transform("min")].groupby("g").agg(a=("a", "median"), q=("q", "max"), age=("age", "first"))
    n_own = d.groupby("g").size()
    own = (last.index.to_numpy(), np.column_stack([last.a, last.q, last.age, n_own.reindex(last.index), np.ones(len(last))]))

    # Building: recency-weighted sums, so a flat's own rows can be subtracted out at lookup time.
    med = pd.Series(a0).groupby(cell).transform("median").to_numpy()
    w = 0.5 ** (age / (HALF_LIFE_MONTHS * 30.4375))
    a_c = np.clip(a0, med - CLIP_LOG, med + CLIP_LOG)
    known = ~np.isnan(floor)
    cols = np.column_stack([np.ones(len(w)), w, w * a_c, w * age, w * known, w * np.nan_to_num(floor)])
    q_i = np.where(np.isnan(q), -1, q).astype("int64")
    own_cell = _compound(gnaf, bed, BED_SLOTS)
    bldg_tables = dict(all=_sums(cell, cols), match=_sums(_compound(cell, q_i, 2), cols),
                       own_all=_sums(own_cell, cols), own_match=_sums(_compound(own_cell, q_i, 2), cols))

    # Block: medians over the last year, from the mesh block outwards.
    mb_ix = pd.Index(flats.mb.dropna().unique(), dtype="object")
    sub_ix = index.groups
    mb = _codes(mb_ix, h.building_key.map(flats.drop_duplicates("building_key").set_index("building_key").mb))
    sub = _codes(sub_ix, h.suburb)
    recent = h.lease_date.to_numpy() > np.datetime64(cutoff - pd.DateOffset(months=BLOCK_WINDOW_MONTHS))
    if recent.sum() < 50:
        recent = np.ones(len(h), bool)  # a thin history uses everything rather than nothing
    block_tiers = []
    for key in (_compound(mb, bed, BED_SLOTS), _compound(sub, bed, BED_SLOTS), sub, bed):
        k = pd.DataFrame({"k": key, "a": a0, "age": age})[recent]
        g = k[k.k >= 0].groupby("k").agg(a=("a", "median"), age=("age", "mean"), n=("a", "size"))
        g = g[g.n >= MIN_BLOCK_ROWS]
        block_tiers.append((g.index.to_numpy(), g.to_numpy()))
    city = np.array([np.median(a0[recent]), age[recent].mean(), recent.sum()])

    baseline = h[h.lease_date > cutoff - pd.DateOffset(months=12)].dropna(subset=["suburb", "bedrooms"])
    return dict(index=index, gnaf_ix=gnaf_ix, bldg_ix=bldg_ix, mb_ix=mb_ix, sub_ix=sub_ix, premium=premium, share=share,
                floor_slope=floor_slope, own=own, bldg=bldg_tables, block_tiers=block_tiers, city=city, baseline=baseline)


def _add_anchors(X: pd.DataFrame, P: dict, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Adds the contract columns. X is one row per target with flat attributes already merged."""
    index = P["index"]
    grow = index.at(X.target_date, X.suburb)
    wait = (X.target_date - cutoff).dt.days.to_numpy(dtype=float)  # ages are measured to target_date
    premium = P["premium"][index.codes(X.suburb)]
    p = X.car_space_resolved.to_numpy(dtype=float)
    p = np.where(np.isnan(p), np.nan, (p > 0).astype(float))
    gnaf, bldg, bed = _codes(P["gnaf_ix"], X.gnaf_id), _codes(P["bldg_ix"], X.building_key), _bed_code(X.bedrooms)
    p_i = np.where(np.isnan(p), -1, p).astype("int64")

    # Own flat. Parking moves it only when both the anchor's and the target's parking are known.
    a, q, age, n, found = _find(P["own"], gnaf).T
    found = found > 0
    q = np.where(found, q, np.nan)
    X["anchor_own"] = np.where(found, np.exp(a + grow + premium * np.nan_to_num(p - q)), np.nan)
    X["anchor_own_age_days"] = np.where(found, age + wait, np.nan)
    X["anchor_own_n"] = n.astype("int64")
    X["anchor_own_parking"] = q
    X["anchor_own_parking_match"] = np.where(np.isnan(p) | np.isnan(q), np.nan, (p == q).astype(float))

    # Building, leaving this flat's own rows out, so it stays independent of anchor_own.
    cell, own_cell = _compound(bldg, bed, BED_SLOTS), _compound(gnaf, bed, BED_SLOTS)
    T = P["bldg"]
    any_ = _find(T["all"], cell) - _find(T["own_all"], own_cell)
    match = _find(T["match"], _compound(cell, p_i, 2)) - _find(T["own_match"], _compound(own_cell, p_i, 2))
    use_match = match[:, 0] >= 0.5
    s = np.where(use_match[:, None], match, any_)
    has = s[:, 0] >= 0.5
    W = np.where(has, s[:, 1], 1.0)
    W = np.where(W > 0, W, 1.0)
    level = s[:, 2] / W + premium * np.where(np.isnan(p), P["share"], p)
    f = pd.to_numeric(X.floor_level, errors="coerce").to_numpy(dtype=float)
    has_f = ~np.isnan(f) & (s[:, 4] > 1e-12)
    mean_f = s[:, 5] / np.where(has_f, s[:, 4], 1.0)
    floor_adj = np.clip(P["floor_slope"] * (np.nan_to_num(f) - mean_f), -MAX_FLOOR_ADJ, MAX_FLOOR_ADJ)
    X["anchor_bldg"] = np.where(has, np.exp(level + np.where(has_f, floor_adj, 0.0) + grow), np.nan)
    X["anchor_bldg_age_days"] = np.where(has, s[:, 3] / W + wait, np.nan)
    X["anchor_bldg_n"] = np.where(has, s[:, 0], 0).round().astype("int64")
    X["anchor_bldg_parking_match"] = np.where(has & ~np.isnan(p), use_match.astype(float), np.nan)

    # Block, then suburb, then bedrooms, then the whole city, so it is never blank.
    mb, sub = _codes(P["mb_ix"], X.mb), _codes(P["sub_ix"], X.suburb)
    level, age, n = (np.full(len(X), P["city"][i], dtype=float) for i in range(3))
    tier = np.full(len(X), len(P["block_tiers"]), dtype="int8")
    keys = (_compound(mb, bed, BED_SLOTS), _compound(sub, bed, BED_SLOTS), sub, bed)
    for t in reversed(range(len(keys))):
        found = _find(P["block_tiers"][t], keys[t])
        hit = found[:, 2] > 0
        level[hit], age[hit], n[hit], tier[hit] = found[hit, 0], found[hit, 1], found[hit, 2], t
    X["anchor_block"] = np.exp(level + premium * np.where(np.isnan(p), P["share"], p) + grow)
    X["anchor_block_age_days"] = age + wait
    X["anchor_block_n"] = n.astype("int64")
    X["anchor_block_level"] = tier

    X["best_anchor"] = X.anchor_own.fillna(X.anchor_bldg).fillna(X.anchor_block)
    X["best_anchor_source"] = np.where(X.anchor_own.notna(), 0, np.where(X.anchor_bldg.notna(), 1, 2)).astype("int8")
    return X


def _block_features(X: pd.DataFrame) -> pd.DataFrame:
    b = load_blocks().drop_duplicates("mb").set_index("mb").reindex(columns=["homes_2021", "homes_2026", "people_2026"])
    b = b.apply(pd.to_numeric, errors="coerce")
    row = b.reindex(X.mb.astype("object")).to_numpy()
    homes21, homes26, people26 = row.T
    with np.errstate(divide="ignore", invalid="ignore"):
        X["block_people_2026"] = people26
        X["block_people_per_home_2026"] = np.where(homes26 > 0, people26 / homes26, np.nan)
        X["block_homes_growth"] = np.where(homes21 > 0, homes26 / homes21 - 1, np.nan)
    return X


def build_features(cutoff_date, target_rows: pd.DataFrame, index_params: IndexParams = DEFAULT_PARAMS) -> pd.DataFrame:
    """target_rows needs gnaf_id, target_date, car_space_resolved. Returns one row per target row, in order.

    Other target_rows columns pass through, except flat attributes, which always come from the input.
    """
    cutoff = pd.Timestamp(cutoff_date)
    assert (target_rows.target_date > cutoff).all(), "targets must be after the cutoff"
    assert cutoff <= C.AS_OF, f"cutoff {cutoff} is after the as-of date"
    leases = load_leases()
    P = memo(leases, ("prepare", cutoff, index_params), lambda: _prepare(leases, cutoff, index_params), maxsize=64)
    flats = flats_asof(cutoff)
    flats["input_car_space_missing"] = flats.car_space.isna().astype("int8")
    flats = flats.drop(columns=["address", "building_address", "car_space"])
    rows = target_rows.reset_index(drop=True)
    rows = rows.drop(columns=[c for c in rows.columns if c in flats.columns and c != "gnaf_id"])
    X = rows.merge(flats, on="gnaf_id", how="left", validate="m:1")
    X["target_date"] = pd.to_datetime(X.target_date)

    for name, park in (("base_with", 1.0), ("base_without", 0.0)):
        X[name] = hier_median(P["baseline"], X.assign(car_space_resolved=park))
    X["attrs_age_days"] = (cutoff - X.attrs_last_seen).dt.days
    X["listed_age_days"] = (cutoff - X.first_ever_listed).dt.days
    X["car_space_resolved"] = X.car_space_resolved.astype(float)

    raw = load_input().set_index("gnaf_id")
    X["floor_level_missing"] = X.floor_level.isna().astype("int8")
    X["bathrooms_missing"] = X.bathrooms.isna().astype("int8")
    X["block_missing"] = X.block_homes_2026.isna().astype("int8")
    # From the unmasked file, so a date hidden because it falls after the cutoff is not flagged as unknown.
    X["attrs_missing"] = X.gnaf_id.map(raw.attrs_last_seen).isna().astype("int8")
    X["listed_missing"] = X.gnaf_id.map(raw.first_ever_listed).isna().astype("int8")

    X = _block_features(X)
    X = _add_anchors(X, P, cutoff)
    X["suburb"] = pd.Categorical(X.suburb, categories=sorted(load_input().suburb.dropna().unique()))
    return X
