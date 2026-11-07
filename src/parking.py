"""Stage 6: parking premium and floor curve from within-building comparisons.

Unit of evidence: one flat in one parking state. Repeat adverts of a flat share its noise, so
counting them separately would overstate precision. Comparisons sit inside a stratum of
building x bedrooms x bathrooms; flats differing in floor are then adjusted by the floor slope.

Time. Rents are deflated to cutoff dollars with a repeat-flat month index, but deflating also
inflates an old premium if that premium did not rise with rents. So the share of the premium
that moves with rents (kappa, 0 = fixed dollars, 1 = proportional) is measured from
same-year comparisons, and estimates are rescaled by it. Everything is in dollars at the cutoff.

Only history_upto(cutoff) and the static floor_level enter, so a fold's table cannot see its test.
"""
import json
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config as C
from .features import flats_asof, history_upto
from .load import load_input
from .validate import score

STRATA = ["building_key", "bedrooms", "bath"]
MIN_CELL_N = 10  # below this many flats, or MIN_CELL_BUILDINGS buildings, a report cell is suppressed
MIN_CELL_BUILDINGS = 5
Z95 = 1.96
TRIM_MADS = 5  # within-stratum rent gaps are clipped here, so one mis-keyed rent cannot set a premium
FLOOR_EDGES = [0, 3, 6, 11, 16, 21, 31, np.inf]
FLOOR_LABELS = ["0-2", "3-5", "6-10", "11-15", "16-20", "21-30", "31+"]
TABLE_COLS = ["level", "building_key", "suburb", "bedrooms", "premium", "premium_raw", "se",
              "n_units", "n_with", "n_without"]
ANCHORS = ("mid", "without", "with")


# ---------- time adjustment ----------

def month_index(h: pd.DataFrame) -> pd.Series:
    """Log rent level by month from repeat adverts of a flat, zero over the last three months.

    Flat fixed effects mean mix changes cannot move it. src/index.py can replace this.
    """
    h = h[h.rent.notna() & h.lease_date.notna()]
    cell = h.gnaf_id.astype(object).where(h.gnaf_id.notna(), h.building_key)
    month = h.lease_date.dt.to_period("M")
    repeat = cell.notna() & cell.groupby(cell).transform("size").gt(1)
    c, _ = pd.factorize(cell[repeat])
    m, months = pd.factorize(month[repeat], sort=True)
    r = np.log(h.rent[repeat].to_numpy(float))
    n_c, n_m = np.bincount(c), np.bincount(m)
    fe = np.zeros(len(months))
    for _ in range(3000):  # alternating projections for the two-way fixed effects
        a = np.bincount(c, r - fe[m]) / n_c
        new = np.bincount(m, r - a[c]) / n_m
        done = np.abs(new - fe).max() < 1e-7
        fe = new
        if done:
            break
    full = pd.period_range(month.min(), month.max(), freq="M")
    s = pd.Series(fe, index=pd.PeriodIndex(months, freq="M")).reindex(full).interpolate(limit_direction="both")
    s = s.rolling(3, center=True, min_periods=1).mean()
    return s - s.tail(3).mean()


def _rows(history: pd.DataFrame) -> pd.DataFrame:
    """Adverts usable for comparison, with parking as 0/1 and rents in cutoff dollars."""
    h = history.dropna(subset=["rent", "lease_date", "building_key", "suburb", "bedrooms", "car_space_resolved"]).copy()
    h["I"] = np.exp(h.lease_date.dt.to_period("M").map(month_index(history)).astype(float))
    h["y"] = h.rent / h.I
    h["car"] = (h.car_space_resolved > 0).astype(float)
    h["bath"] = h.bathrooms.fillna(-1)
    h["bedrooms"] = h.bedrooms.astype(float)
    h["suburb"] = h.groupby("building_key").suburb.transform("first")
    row_id = pd.Series("row" + h.index.astype(str), index=h.index)
    h["flat"] = h.gnaf_id.astype(object).where(h.gnaf_id.notna(), row_id)
    if "floor_level" not in h:
        h["floor_level"] = np.nan
    return h


def _units(rows: pd.DataFrame, kappa: float) -> pd.DataFrame:
    """One row per flat and parking state. s is what $1 of cutoff-date effect looks like in y."""
    r = rows.assign(s=kappa + (1 - kappa) / rows.I)
    return r.groupby(["flat", "car"] + STRATA, as_index=False, observed=True).agg(
        y=("y", "mean"), s=("s", "mean"), suburb=("suburb", "first"), floor_level=("floor_level", "first"))


def _within(df: pd.DataFrame, keys: list, x: pd.Series, y: pd.Series) -> pd.DataFrame:
    """Demean x and y inside strata and keep the rows that carry a comparison."""
    g = [df[k] for k in keys]
    out = df.assign(xt=x - x.groupby(g).transform("mean"), yt=y - y.groupby(g).transform("mean"))
    out = out[out.xt.abs() > 1e-9].copy()
    if len(out):
        cap = TRIM_MADS * 1.4826 * out.yt.abs().median()
        out["yt"] = out.yt.clip(-cap, cap)
    return out.assign(sxx=out.xt ** 2, sxy=out.xt * out.yt)


def _pooled(w: pd.DataFrame) -> dict:
    """Slope over comparison rows with a standard error clustered on building."""
    if not len(w) or w.sxx.sum() == 0:
        return dict(est=np.nan, se=np.nan, n=0, n_buildings=0, sxx=0.0)
    b = w.groupby("building_key", observed=True)[["sxx", "sxy"]].sum()
    est = b.sxy.sum() / b.sxx.sum()
    k = len(b)
    se = np.sqrt(k / (k - 1) * ((b.sxy - est * b.sxx) ** 2).sum()) / b.sxx.sum() if k > 1 else np.nan
    return dict(est=float(est), se=float(se), n=len(w), n_buildings=k, sxx=float(b.sxx.sum()))


def _by_year(rows: pd.DataFrame, x: str, keys: list) -> pd.DataFrame:
    """Same-year effect of x in that year's dollars, so no deflator touches it."""
    r = rows.assign(year=rows.lease_date.dt.year)
    level = r.groupby("year").I.transform("mean")
    w = _within(r, keys + ["year"], r[x].astype(float), r.y * level)
    out = pd.DataFrame([{"year": y, **_pooled(g)} for y, g in w.groupby("year")])
    if len(out):
        out["level"] = out.year.map(r.groupby("year").I.mean())
    return out


def _kappa(by_year: pd.DataFrame) -> float:
    """Share of an effect that moves with rents: fit effect = a + b * level, then b / (a + b) at the cutoff."""
    d = by_year[by_year.sxx > 0] if len(by_year) else by_year
    if len(d) < 3 or d.level.max() - d.level.min() < 0.05:
        return 1.0
    b, a = np.polyfit(d.level, d.est, 1, w=np.sqrt(d.sxx))
    return float(np.clip(b / (a + b), 0, 1)) if a + b > 0 else 1.0


# ---------- floor curve ----------

def _floor_fit(rows: pd.DataFrame) -> dict | None:
    """$/week per extra floor at the cutoff, inside building x bedrooms x bathrooms x parking."""
    r = rows[rows.floor_level.notna()]
    if not len(r):
        return None
    keys = STRATA + ["car"]
    by_year = _by_year(r, "floor_level", keys)
    kappa = _kappa(by_year)
    u = _units(r, kappa)
    g = [u[k] for k in keys]
    s_bar = u.s.groupby(g).transform("mean")  # one scale per stratum: time spread must not pose as floor spread
    w = _within(u, keys, u.floor_level * s_bar, u.y)
    return dict(kappa=kappa, by_year=by_year, slope=_pooled(w), units=u.assign(s_bar=s_bar), keys=keys)


def _floor_bands(fit: dict) -> pd.DataFrame:
    """Rent of each floor band against the lowest band in the same stratum, clustered on building."""
    u = fit["units"].copy()
    u["band"] = pd.cut(u.floor_level, FLOOR_EDGES, right=False, labels=FLOOR_LABELS)
    keys = fit["keys"]
    g = [u[k] for k in keys]
    u = u[u.band.astype(str).groupby(g).transform("nunique") > 1]
    if not len(u):
        return pd.DataFrame(columns=["band", "floor_mean", "n", "n_buildings", "premium", "se", "per_floor"])
    g = [u[k] for k in keys]
    present = [b for b in FLOOR_LABELS if (u.band == b).any()]
    D = pd.DataFrame({b: (u.band == b) * u.s_bar for b in present[1:]}, index=u.index)
    X = (D - D.groupby(g).transform("mean")).to_numpy(float)
    y = (u.y - u.y.groupby(g).transform("mean")).to_numpy(float)
    cap = TRIM_MADS * 1.4826 * np.median(np.abs(y))
    y = np.clip(y, -cap, cap)
    bread = np.linalg.pinv(X.T @ X)
    beta = bread @ X.T @ y
    scores = pd.DataFrame(X * (y - X @ beta)[:, None]).groupby(u.building_key.to_numpy()).sum().to_numpy()
    k = len(scores)
    cov = bread @ (scores.T @ scores) @ bread * k / max(k - 1, 1)
    out = u.groupby("band", observed=True).agg(floor_mean=("floor_level", "mean"), n=("y", "size"),
                                               n_buildings=("building_key", "nunique")).reindex(present).reset_index()
    out["premium"] = np.r_[0.0, beta]
    out["se"] = np.r_[0.0, np.sqrt(np.diag(cov))]
    out["per_floor"] = out.premium.diff() / out.floor_mean.diff()
    return out


# ---------- premium table ----------

def _shrink(beta: pd.Series, se2: pd.Series, prior: pd.Series) -> tuple[pd.Series, float]:
    """Empirical Bayes toward prior, between-cell variance by DerSimonian-Laird."""
    w = 1 / se2
    q = (w * (beta - prior) ** 2).sum()
    denom = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (q - (len(beta) - 1)) / denom) if denom > 0 else 0.0
    return prior + tau2 / (tau2 + se2) * (beta - prior), tau2


def _cells(w: pd.DataFrame, keys: list) -> pd.DataFrame:
    if not keys:
        w = w.assign(_all=0)
        keys = ["_all"]
    c = w.groupby(keys, as_index=False, observed=True).agg(
        sxx=("sxx", "sum"), sxy=("sxy", "sum"), n_units=("car", "size"), n_with=("car", "sum"))
    c["n_with"] = c.n_with.astype(int)
    c["n_without"] = c.n_units - c.n_with
    c["premium_raw"] = c.sxy / c.sxx
    return c


@dataclass
class PremiumFit:
    cutoff: pd.Timestamp
    table: pd.DataFrame
    comparisons: pd.DataFrame  # comparison rows; row-level, never written out
    by_year: pd.DataFrame
    floor: dict | None
    kappa: float
    sigma: float
    tau_suburb: float
    tau_building: float


def _empty_table() -> pd.DataFrame:
    row = dict(level="all", building_key=pd.NA, suburb=pd.NA, bedrooms=np.nan, premium=0.0, premium_raw=np.nan,
               se=np.nan, n_units=0, n_with=0, n_without=0)
    return pd.DataFrame([row])[TABLE_COLS]


def fit_from_history(history: pd.DataFrame, cutoff=None) -> PremiumFit:
    """Fit from a history frame. Needs the columns of history_upto; floor_level is optional."""
    rows = _rows(history)
    floor = _floor_fit(rows)
    by_year = _by_year(rows, "car", STRATA)
    kappa = _kappa(by_year)

    u = _units(rows, kappa)
    g = [u[k] for k in STRATA]
    if floor is not None and np.isfinite(floor["slope"]["est"]):
        gap = u.floor_level - u.floor_level.groupby(g).transform("mean")
        u["y"] = u.y - floor["slope"]["est"] * u.s * gap.fillna(0)
    s_with = u.s.where(u.car == 1).groupby(g).transform("mean")  # the with arm's premium sits at this scale
    w = _within(u, STRATA, u.car * s_with.fillna(0), u.y)
    w = w[w.groupby(STRATA, observed=True).car.transform("nunique") == 2]
    if not len(w):
        return PremiumFit(cutoff, _empty_table(), w, by_year, floor, kappa, np.nan, 0.0, 0.0)

    sb = _cells(w, ["suburb", "bedrooms"])
    beta_sb = w.merge(sb[["suburb", "bedrooms", "premium_raw"]], on=["suburb", "bedrooms"]).set_index(w.index).premium_raw
    dof = len(w) - w.groupby(STRATA, observed=True).ngroups - len(sb)
    sigma2 = float(((w.yt - beta_sb * w.xt) ** 2).sum() / max(dof, 1))

    top = _cells(w, [])
    top_post = float(top.premium_raw.iloc[0])
    bed = _cells(w, ["bedrooms"])
    bed["post"], _ = _shrink(bed.premium_raw, sigma2 / bed.sxx, pd.Series(top_post, index=bed.index))
    sb = sb.merge(bed[["bedrooms", "post"]].rename(columns={"post": "prior"}), on="bedrooms")
    sb["post"], tau2_s = _shrink(sb.premium_raw, sigma2 / sb.sxx, sb.prior)
    bb = _cells(w, ["building_key", "bedrooms", "suburb"])
    bb = bb.merge(sb[["suburb", "bedrooms", "post"]].rename(columns={"post": "prior"}), on=["suburb", "bedrooms"])
    bb["post"], tau2_b = _shrink(bb.premium_raw, sigma2 / bb.sxx, bb.prior)

    top["post"] = top_post
    parts = [top.assign(level="all"), bed.assign(level="bedrooms"), sb.assign(level="suburb"), bb.assign(level="building")]
    table = pd.concat(parts, ignore_index=True)
    table["premium"] = table.post.clip(lower=0)  # floored last, so shrinkage still sees negative evidence
    table["se"] = np.sqrt(sigma2 / table.sxx)
    table = table.reindex(columns=TABLE_COLS)
    return PremiumFit(cutoff, table, w, by_year, floor, kappa, float(np.sqrt(sigma2)),
                      float(np.sqrt(tau2_s)), float(np.sqrt(tau2_b)))


def fit_premium(cutoff) -> PremiumFit:
    cutoff = pd.Timestamp(cutoff)
    floors = flats_asof(cutoff)[["gnaf_id", "floor_level"]]
    return fit_from_history(history_upto(cutoff).merge(floors, on="gnaf_id", how="left"), cutoff)


def estimate_premium(cutoff) -> pd.DataFrame:
    """Premium table, $/week at the cutoff. `level` is building, suburb, bedrooms or all.

    Building rows are keyed building_key x bedrooms and exist only where a building has both arms.
    """
    return fit_premium(cutoff).table


def premium_for(flats_df: pd.DataFrame, table: pd.DataFrame) -> pd.Series:
    """Premium per flat from its most specific cell. flats_df needs building_key, suburb, bedrooms. Never NaN."""
    f = flats_df[["building_key", "suburb", "bedrooms"]].reset_index(drop=True)
    f = f.assign(bedrooms=pd.to_numeric(f.bedrooms, errors="coerce").astype(float))
    out = pd.Series(np.nan, index=f.index)
    for level, keys in (("building", ["building_key", "bedrooms"]), ("suburb", ["suburb", "bedrooms"]), ("bedrooms", ["bedrooms"])):
        t = table.loc[table.level == level, keys + ["premium"]].astype({"bedrooms": float})
        out = out.fillna(f.merge(t, on=keys, how="left")["premium"].where(f[keys].notna().all(axis=1)))
    top = table.loc[table.level == "all", "premium"]
    out = out.fillna(float(top.iloc[0]) if len(top) else 0.0)
    out.index = flats_df.index
    return out.rename("parking_premium")


# ---------- validation ----------

def fold_premium(fold: str, rows: pd.DataFrame, table: pd.DataFrame | None = None) -> pd.Series:
    """Premium for rows with a gnaf_id, from the table fitted at the fold's cutoff."""
    table = estimate_premium(C.FOLDS[fold]["cutoff"]) if table is None else table
    keys = rows[["gnaf_id"]].merge(load_input()[["gnaf_id", "building_key", "suburb", "bedrooms"]], on="gnaf_id", how="left")
    keys.index = rows.index
    return premium_for(keys, table)


def apply_premium(df: pd.DataFrame, premium: pd.Series, anchor: str = "mid") -> pd.DataFrame:
    """Replace the model's with/without gap by `premium`.

    anchor: without keeps rent_without, with keeps rent_with, mid keeps their midpoint. Mid changes
    only the gap, so it isolates the premium from any level difference between the model's two passes.
    """
    w, wo = df["rent_with_car_space"], df["rent_without_car_space"]
    base = {"without": wo, "with": w - premium, "mid": (w + wo) / 2 - premium / 2}[anchor]
    return df.assign(rent_without_car_space=base, rent_with_car_space=base + premium)


def _segments(df: pd.DataFrame) -> dict:
    adv, inp = df["advert_car_space"].astype(float), df["input_car_space"].astype(float)
    park = adv.fillna(inp)
    return {"all": park.notna(), "parking": park > 0, "no_parking": park == 0,
            "flipped": adv.notna() & inp.notna() & ((adv > 0) != (inp > 0))}


def compare_premium_methods(df: pd.DataFrame, premium: pd.Series, anchors=ANCHORS) -> pd.DataFrame:
    """Score a predictor's own gap against `premium` on test adverts, with src.validate.score.

    df needs the columns of score(), with the predictor's two passes in rent_with_car_space and
    rent_without_car_space. Returns a row per method and segment, d_mae being the change in median
    absolute error against the predictor's own gap (negative is better).

    Segments: all, parking and no_parking follow the scored side; flipped is where the advert
    contradicts the input flag, the only adverts that score the side a flat's input does not claim.
    """
    methods = {"model": df, **{f"premium_{a}": apply_premium(df, premium, a) for a in anchors}}
    seg = _segments(df)
    out = []
    for name, d in methods.items():
        for s, mask in seg.items():
            gap = d.loc[mask, "rent_with_car_space"] - d.loc[mask, "rent_without_car_space"]
            out.append(dict(method=name, segment=s, **score(d[mask]), median_gap=float(gap.median()),
                            share_negative_gap=float((gap < 0).mean())))
    res = pd.DataFrame(out)
    res["d_mae"] = res.mae - res.groupby("segment").mae.transform("first")
    return res


MIN_FLIPPED = 30  # fewer adverts than this cannot separate two premiums
CLEAR_WIN = 5.0  # dollars of median error on flipped adverts


def pick_premium(comparison: pd.DataFrame) -> tuple[str, str]:
    """Choose 'within_building' or 'two_pass' from compare_premium_methods output on Fold B.

    Within-building is the default: it is a direct measurement, while the two-pass gap is compressed because
    anchors stay fixed between passes. Flipped adverts are few, so two-pass must win them clearly to replace it.
    """
    c = comparison.set_index(["method", "segment"])
    if c.loc[("model", "flipped"), "n"] < MIN_FLIPPED:
        return "within_building", "default: too few flipped adverts on Fold B"
    d = c.loc[("premium_mid", "flipped"), "d_mae"]
    if d > CLEAR_WIN:
        return "two_pass", f"two-pass beat within-building by ${d:.2f} on Fold B flipped adverts"
    return "within_building", f"default: two-pass margin on Fold B flipped adverts was ${d:.2f}, under ${CLEAR_WIN:.0f}"


# ---------- report aggregates ----------

def _cell(name: str, value, p: dict) -> dict:
    """One report cell. Small cells keep their label and lose every number, n included."""
    small = p["n"] < MIN_CELL_N or p["n_buildings"] < MIN_CELL_BUILDINGS or not np.isfinite(p["se"])
    if small:
        return {name: value, "suppressed": True, "premium": None, "low": None, "high": None, "n": None, "n_buildings": None}
    return {name: value, "suppressed": False, "premium": round(float(p["est"]), 2),
            "low": round(float(p["est"] - Z95 * p["se"]), 2), "high": round(float(p["est"] + Z95 * p["se"]), 2),
            "n": int(p["n"]), "n_buildings": int(p["n_buildings"])}


def _head(fit: PremiumFit) -> dict:
    """The stamp src/report.py requires on every file, built here because that module is on another branch."""
    return {"provenance": {"mode": "mock" if C.MOCK else "real",
                           "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
            "cutoff": str(fit.cutoff.date()) if fit.cutoff is not None else None, "unit": "$/week at the cutoff"}


def parking_report(fit: PremiumFit) -> dict:
    """Pooled, unshrunk, unfloored estimates: flooring cells first would bias an average upward."""
    w = fit.comparisons
    return {
        **_head(fit),
        "method": "flats with vs without parking inside building x bedrooms x bathrooms; 95% CI clustered on building; "
                  "n counts flats in those comparisons (adverts in by_year)",
        "min_cell": {"flats": MIN_CELL_N, "buildings": MIN_CELL_BUILDINGS},
        "share_moving_with_rents": round(fit.kappa, 3),
        "overall": _cell("scope", "all", _pooled(w)),
        "by_bedrooms": [_cell("bedrooms", int(k), _pooled(g)) for k, g in w.groupby("bedrooms")],
        "by_suburb": [_cell("suburb", k, _pooled(g)) for k, g in w.groupby("suburb")],
        "by_year": [_cell("year", int(r["year"]), r.to_dict()) for _, r in fit.by_year.iterrows()],
    }


def floor_report(fit: PremiumFit) -> dict:
    """Curve premiums are against reference_floor, the typical floor of the lowest band; `floor` is each band's."""
    f = fit.floor
    bands = _floor_bands(f) if f is not None else pd.DataFrame()
    if not len(bands):
        return {**_head(fit), "per_floor_premium": None, "reference_floor": None, "curve": []}
    curve = []
    for _, r in bands.iterrows():
        c = _cell("floor", int(round(r.floor_mean)), dict(est=r.premium, se=r.se, n=r.n, n_buildings=r.n_buildings))
        c["band"] = r.band
        c["per_extra_floor"] = None if c["suppressed"] or pd.isna(r.per_floor) else round(float(r.per_floor), 2)
        curve.append(c)
    slope = _cell("scope", "all", f["slope"])
    return {**_head(fit), "method": "flats on different floors inside building x bedrooms x bathrooms x parking; "
                                    "95% CI clustered on building; n counts flats",
            "min_cell": {"flats": MIN_CELL_N, "buildings": MIN_CELL_BUILDINGS},
            "share_moving_with_rents": round(f["kappa"], 3),
            "per_floor_premium": slope["premium"], "per_floor_low": slope["low"], "per_floor_high": slope["high"],
            "per_floor_n": slope["n"], "reference_floor": int(round(bands.floor_mean.iloc[0])), "curve": curve}


def write_reports(fit: PremiumFit) -> dict:
    """Write parking.json and floor.json under C.OUT_DIR. Both hold aggregates only."""
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, rep in (("parking.json", parking_report(fit)), ("floor.json", floor_report(fit))):
        paths[name] = C.OUT_DIR / name
        paths[name].write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    return paths
