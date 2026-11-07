"""Stage 3: monthly log-rent index, fitted only on rows dated <= cutoff.

One penalised solve per curve, in log rent, with a month dummy per calendar month:

    min  sum_pairs w (dlog_rent - (I[t2] - I[t1]))^2      repeat leases of one flat
       + median_weight * sum_t n_t (m_t - I[t])^2          mix-adjusted monthly medians
       + smooth * |second difference of (I - prior)|^2     carries sparse and empty months
       + shrink * |I - prior|^2                            thin suburbs fall back on the prior

Repeat pairs dominate where they exist. Where they are thin the medians take over, and
where both are thin a suburb reverts to the all-suburb curve. Nothing here needs a minimum
row count, so a suburb or month with no rows is not a special case.

Curves are normalised to 0 at the cutoff, so a value is "log rent change since the cutoff".
Past the last month the curve follows the trend up to the cutoff, damped by IndexParams.
Tune trend_months, damping and max_horizon_months on Fold B only.
"""
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pandas as pd

MONTH_DAYS = 30.4375


@dataclass(frozen=True)
class IndexParams:
    trend_months: int = 12            # trend = change over this span; 12 cancels seasonality
    damping: float = 1.0              # per-month decay of the projected trend; 1 = none, 0 = flat
    max_horizon_months: float = 6.0   # no further projection beyond this
    smooth: float = 30.0
    shrink: float = 5.0
    median_weight: float = 0.02
    use_index: bool = True            # False gives a flat index, for "before index" comparisons


DEFAULT_PARAMS = IndexParams(damping=0.0, trend_months=6)  # tuned on real Fold B: holding the index flat past the cutoff won

_MEMO: OrderedDict = OrderedDict()


def memo(source: pd.DataFrame, key: tuple, make, maxsize: int = 64):
    """Cache per source frame. The entry holds the frame, so its id cannot be reused,
    and a patched or edited copy of the leases is a miss rather than a stale hit."""
    k = (id(source),) + key
    if k in _MEMO:
        _MEMO.move_to_end(k)
        return _MEMO[k][1]
    value = make()
    _MEMO[k] = (source, value)
    while len(_MEMO) > maxsize:
        _MEMO.popitem(last=False)
    return value


def _days(dates) -> np.ndarray:
    """Days since the epoch as floats; NaT becomes NaN."""
    return (pd.DatetimeIndex(pd.to_datetime(dates)) - pd.Timestamp(0)).total_seconds().to_numpy() / 86_400


class RentIndex:
    """Log index by suburb, with one extra row (the last) for all suburbs."""

    def __init__(self, cutoff, groups, knots, curves, slopes, params, n_pairs=0):
        self.cutoff, self.params, self.n_pairs = pd.Timestamp(cutoff), params, n_pairs
        self.groups = pd.Index(groups, dtype="object")
        self.knots, self.curves, self.slopes = knots, curves, slopes
        here = self._raw(np.full(len(curves), _days([cutoff])[0]), np.arange(len(curves)))
        self.curves = curves - here[:, None]

    def codes(self, suburb) -> np.ndarray:
        """Row of self.curves for each suburb. Unknown or blank suburbs use the all-suburb row."""
        c = self.groups.get_indexer(pd.Series(suburb).astype("object").to_numpy())
        return np.where(c < 0, len(self.groups), c)

    def _raw(self, days, rows):
        k, cv, p = self.knots, self.curves, self.params
        hi = np.clip(np.searchsorted(k, days), 1, len(k) - 1) if len(k) > 1 else np.zeros(len(days), int)
        lo = np.maximum(hi - 1, 0)
        span = np.where(k[hi] > k[lo], k[hi] - k[lo], 1.0)
        f = np.clip((days - k[lo]) / span, 0, 1)
        out = cv[rows, lo] * (1 - f) + cv[rows, hi] * f
        h = np.clip((days - k[-1]) / MONTH_DAYS, 0, p.max_horizon_months)
        phi = p.damping
        steps = h if phi >= 1 else (phi * (1 - phi ** h) / (1 - phi) if phi > 0 else np.zeros_like(h))
        return out + self.slopes[rows] * steps

    def at(self, dates, suburb=None) -> np.ndarray:
        """Log rent change from the cutoff to each date, projected past the cutoff."""
        days = _days(dates)
        rows = np.full(len(days), len(self.groups)) if suburb is None else self.codes(suburb)
        return self._raw(days, rows)


def _repeat_pairs(d: pd.DataFrame) -> pd.DataFrame:
    """Consecutive leases of one flat in different months, like for like on beds and parking."""
    p = d[d.gnaf_id.notna()].sort_values(["gnaf_id", "lease_date"], kind="stable")
    prev = p.shift()
    keep = p.gnaf_id.eq(prev.gnaf_id).fillna(False).to_numpy(dtype=bool) & (p._m != prev._m).to_numpy()
    keep &= ~(p.bedrooms.notna() & prev.bedrooms.notna() & (p.bedrooms != prev.bedrooms)).to_numpy()
    keep &= ~(p.car_space_resolved.notna() & prev.car_space_resolved.notna()
              & (p.car_space_resolved != prev.car_space_resolved)).to_numpy()
    out = pd.DataFrame({"t1": prev._m[keep].astype(int), "t2": p._m[keep], "y": (p._y - prev._y)[keep], "g": p._g[keep]})
    return out.assign(w=1 / (1 + (out.t2 - out.t1) / 24))  # long gaps carry more unrelated change


def _monthly_medians(rows: pd.DataFrame, prior: np.ndarray, T: int):
    """Median log rent by month after removing each suburb x bedrooms cell's level."""
    r = rows._y - prior[rows._m.to_numpy()]
    fe = r.groupby([rows._g, rows._b], dropna=False).transform("median")
    g = (rows._y - fe).groupby(rows._m).agg(["median", "size"])
    m, n = np.zeros(T), np.zeros(T)
    m[g.index.to_numpy()], n[g.index.to_numpy()] = g["median"].to_numpy(), g["size"].to_numpy()
    return m, n


def _solve(pairs: pd.DataFrame, m, n, prior, p: IndexParams, shrink: float) -> np.ndarray:
    T = len(prior)
    t1, t2, w = pairs.t1.to_numpy(), pairs.t2.to_numpy(), pairs.w.to_numpy()
    y = pairs.y.to_numpy() - (prior[t2] - prior[t1])
    A, b = np.zeros((T, T)), np.zeros(T)
    np.add.at(A, (t1, t1), w)
    np.add.at(A, (t2, t2), w)
    np.add.at(A, (t1, t2), -w)
    np.add.at(A, (t2, t1), -w)
    np.add.at(b, t2, w * y)
    np.add.at(b, t1, -w * y)
    A[np.diag_indices(T)] += p.median_weight * n + shrink
    b += p.median_weight * n * (m - prior)
    if T >= 3:
        D = np.diff(np.eye(T), 2, axis=0)
        A += p.smooth * D.T @ D
    return prior + np.linalg.solve(A, b)


def _slope(curve: np.ndarray, k: int) -> float:
    """Monthly trend: last three months against the same three k months earlier."""
    T = len(curve)
    if T >= k + 3:
        return float((curve[-3:].mean() - curve[-3 - k:T - k].mean()) / k)
    return float((curve[-1] - curve[0]) / (T - 1)) if T > 1 else 0.0


def fit_index(hist: pd.DataFrame, cutoff, params: IndexParams = DEFAULT_PARAMS) -> RentIndex:
    """hist needs lease_date, rent, gnaf_id, suburb, bedrooms, car_space_resolved, all dated <= cutoff."""
    cutoff = pd.Timestamp(cutoff)
    assert len(hist), "no leases on or before the cutoff"
    assert hist.lease_date.max() <= cutoff, "index given rows after its cutoff"
    first, last = hist.lease_date.min().to_period("M"), cutoff.to_period("M")
    T = (last - first).n + 1
    starts = pd.period_range(first, last, freq="M").to_timestamp()
    knots = _days(starts) + starts.days_in_month.to_numpy() / 2
    knots[-1] = (_days(starts[-1:])[0] + _days([cutoff])[0]) / 2  # the last month may be partial

    groups = pd.Index(sorted(hist.suburb.dropna().unique()), dtype="object")
    G = len(groups)
    if not params.use_index:
        return RentIndex(cutoff, groups, knots, np.zeros((G + 1, T)), np.zeros(G + 1), params)

    d = pd.DataFrame({
        "gnaf_id": hist.gnaf_id, "lease_date": hist.lease_date, "bedrooms": hist.bedrooms,
        "car_space_resolved": hist.car_space_resolved, "_y": np.log(hist.rent.astype(float)),
        "_m": (hist.lease_date.dt.year - first.year) * 12 + hist.lease_date.dt.month - first.month,
        "_g": groups.get_indexer(hist.suburb.astype("object").to_numpy()), "_b": hist.bedrooms})
    pairs = _repeat_pairs(d)

    curve = np.zeros(T)
    for step in range(2):
        m, n = _monthly_medians(d, curve, T)
        curve = _solve(pairs, m, n, np.zeros(T), params, 1e-3)
        if step == 0 and len(pairs):  # drop gross mismatches, such as a wrong flat or a refit
            e = pairs.y - (curve[pairs.t2.to_numpy()] - curve[pairs.t1.to_numpy()])
            e = (e - e.median()).abs()
            pairs = pairs[e <= max(4 * 1.4826 * e.median(), 0.25)]

    curves = np.empty((G + 1, T))
    curves[-1] = curve
    by_group = dict(tuple(pairs.groupby("g")))
    empty = pairs.iloc[:0]
    for g in range(G):
        rows = d[d._g == g]
        m, n = _monthly_medians(rows, curve, T)
        curves[g] = _solve(by_group.get(g, empty), m, n, curve, params, params.shrink)
    slopes = np.array([_slope(c, params.trend_months) for c in curves])
    return RentIndex(cutoff, groups, knots, curves, slopes, params, n_pairs=len(pairs))


def rent_index(leases: pd.DataFrame, cutoff, params: IndexParams = DEFAULT_PARAMS) -> RentIndex:
    """The index as known at cutoff, cached per cutoff. The date filter is applied here."""
    cutoff = pd.Timestamp(cutoff)
    return memo(leases, ("index", cutoff, params), lambda: fit_index(leases[leases.lease_date <= cutoff], cutoff, params))
