"""Shared pieces of the model zoo: windows, the raw-last-rent features, matrices. Scoring lives in local/zoo/stack.py only."""
import numpy as np, pandas as pd
from src import config as C, features as F, models as M

ZOO = C.OUT_DIR / "zoo"
PANEL = ZOO / "panel.parquet"
FINAL = ZOO / "final.parquet"
PRED = ZOO / "preds"
WINDOWS = {"w6": "2025-03-31", "w5": "2025-05-31", "w4": "2025-07-31", "w3": "2025-09-30", "w2": "2025-11-30", "w1": "2026-01-31",
           "B": "2026-03-31", "A": "2026-05-31", "F": "2026-07-31"}
RAW_COLS = ["raw_prev_ratio", "prev_m50", "prev_gap_days"]
V2_COLS = ["own_cagr", "prev_same", "own_bed_match", "own_bldg_gap", "own_block_gap", "bldg_block_gap", "raw_bldg_gap"]
N_ORIGINS = 12


def add_raw(df: pd.DataFrame, leases: pd.DataFrame) -> pd.DataFrame:
    """Last raw rent of each flat on or before the row's origin. Uses nothing dated after the origin."""
    h = leases.dropna(subset=["gnaf_id"])[["gnaf_id", "lease_date", "rent"]].rename(columns={"rent": "prev_rent", "lease_date": "prev_date"})
    h = h.assign(gnaf_id=h.gnaf_id.astype(object)).sort_values("prev_date")
    x = df.reset_index(drop=True).assign(_i=lambda d: np.arange(len(d)), gnaf_id=lambda d: d.gnaf_id.astype(object)).sort_values("origin")
    x = pd.merge_asof(x, h, left_on="origin", right_on="prev_date", by="gnaf_id", direction="backward", allow_exact_matches=True)
    x = x.sort_values("_i").drop(columns="_i").reset_index(drop=True)
    assert (x.prev_date.dropna() <= x.origin[x.prev_date.notna()]).all()
    x["raw_prev_ratio"] = np.log(x.prev_rent / x.best_anchor)
    x["prev_m50"] = (x.prev_rent % 50 == 0).astype(float).where(x.prev_rent.notna())
    x["prev_gap_days"] = (x.origin - x.prev_date).dt.days.astype(float)
    return x.drop(columns=["prev_rent", "prev_date"])


def add_v2(df: pd.DataFrame, leases: pd.DataFrame) -> pd.DataFrame:
    """A flat's last two adverts on or before the origin, and gaps between anchors. Nothing dated after the origin."""
    h = leases.dropna(subset=["gnaf_id"])[["gnaf_id", "lease_date", "rent", "bedrooms"]].sort_values("lease_date")
    h = h.assign(gnaf_id=h.gnaf_id.astype(object))
    g = h.groupby("gnaf_id")
    h = h.assign(r2=g.rent.shift(), d2=g.lease_date.shift()).rename(columns={"rent": "r1", "lease_date": "d1", "bedrooms": "b1"})
    x = df.reset_index(drop=True).assign(_i=lambda d: np.arange(len(d)), gnaf_id=lambda d: d.gnaf_id.astype(object)).sort_values("origin")
    x = pd.merge_asof(x, h, left_on="origin", right_on="d1", by="gnaf_id", direction="backward", allow_exact_matches=True)
    x = x.sort_values("_i").drop(columns="_i").reset_index(drop=True)
    assert (x.d1.dropna() <= x.origin[x.d1.notna()]).all()
    yrs = ((x.d1 - x.d2).dt.days / 365.25).clip(lower=.25)
    x["own_cagr"] = np.log(x.r1 / x.r2) / yrs
    x["prev_same"] = (x.r1 == x.r2).astype(float).where(x.r2.notna())
    x["own_bed_match"] = (x.b1 == pd.to_numeric(x.bedrooms, errors="coerce")).astype(float).where(x.b1.notna())
    x["own_bldg_gap"] = np.log(x.anchor_own / x.anchor_bldg)
    x["own_block_gap"] = np.log(x.anchor_own / x.anchor_block)
    x["bldg_block_gap"] = np.log(x.anchor_bldg / x.anchor_block)
    x["raw_bldg_gap"] = np.log(x.r1 / x.anchor_bldg)
    return x.drop(columns=["r1", "d1", "b1", "r2", "d2"])


def load():
    p = pd.read_parquet(PANEL); f = pd.read_parquet(FINAL)
    for d in (p, f):
        d["origin"], d["target_date"] = pd.to_datetime(d.origin), pd.to_datetime(d.target_date)
    return p, f


def columns(panel, fs):
    cols, _ = M.resolve_features(panel, dict(feature_cols=F.FEATURE_COLS))
    return list(dict.fromkeys(cols + {"base": [], "raw": RAW_COLS, "v2": RAW_COLS + V2_COLS}[fs]))  # main already lists the raw columns


def split(panel, final, window, n=N_ORIGINS):
    cut = pd.Timestamp(WINDOWS[window])
    tr = M.train_rows(panel, cut, n)
    tr = tr[tr.best_anchor > 0]
    assert tr.target_date.max() <= cut
    te = final if window == "F" else panel[panel.origin == cut]
    return tr.reset_index(drop=True), te.reset_index(drop=True)


LOG1P = ["flats_in_building", "block_new_homes_2021_2026", "block_homes_2026", "block_people_2026", "anchor_own_n", "anchor_bldg_n", "anchor_block_n",
         "anchor_own_age_days", "anchor_bldg_age_days", "anchor_block_age_days", "prev_gap_days"]


class Dense:
    """Numeric matrix for models that need one: anchors as log ratios, median fill plus missing flags, suburb one-hot or ordinal."""
    def __init__(self, tr, cols, onehot=True, scale=True):
        self.cols, self.onehot, self.scale = [c for c in cols if c != "suburb"], onehot, scale
        self.subs = sorted(tr.suburb.astype(object).dropna().unique())
        a = self._base(tr)
        self.med = a.median(); self.flags = [c for c in a if a[c].isna().any()]
        b = self._fill(a)
        self.keep = [c for c in b if b[c].nunique() > 1]
        b = b[self.keep]; self.mu, self.sd = b.mean(), b.std().replace(0, 1)

    def _base(self, d):
        x = pd.DataFrame({c: pd.to_numeric(d[c], errors="coerce").astype(float) for c in self.cols})
        ba = np.log(d.best_anchor.astype(float))
        for c in ("anchor_own", "anchor_bldg", "anchor_block"):
            x[c] = np.log(x[c]) - ba
        x["best_anchor"] = ba
        for c in LOG1P:
            if c in x:
                x[c] = np.log1p(x[c].clip(lower=0))
        return x

    def _fill(self, a):
        b = a.fillna(self.med)
        for c in self.flags:
            b[c + "_na"] = a[c].isna().astype(float)
        return b

    def __call__(self, d, car=None):
        b = self._fill(self._base(d))[self.keep].copy()
        if car is not None:
            b["car_space_resolved"] = float(car)
        if self.scale:
            b = (b - self.mu) / self.sd
        s = d.suburb.astype(object)
        if self.onehot:
            for k in self.subs:
                b["sub_" + k] = (s == k).astype(float).to_numpy()
        else:
            b["suburb_code"] = pd.Series(s.to_numpy()).map({k: i for i, k in enumerate(self.subs)}).fillna(-1).to_numpy()
        return b.astype(np.float32)
