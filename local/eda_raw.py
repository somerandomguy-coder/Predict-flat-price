"""Raw-data EDA aimed at model choice. Rent-based results use adverts dated <= EDA_END, so Folds A and B stay unread.
Prints aggregates only. Run from the repo root with PYTHONPATH=."""
import numpy as np, pandas as pd
from src import config as C
from src.load import load_input, load_leases, load_blocks

EDA_END = pd.Timestamp("2026-03-31")
pd.set_option("display.width", 180, "display.max_rows", 200, "display.max_columns", 40)
def h(t): print(f"\n## {t}", flush=True)

F = load_input(); Lall = load_leases()
raw = pd.read_csv(C.LEASES_CSV, low_memory=False, usecols=["gnaf_id", "listing_date", "was_leased", "property_type", "garage_spaces"])
Lall = Lall.copy()
L = Lall[Lall.lease_date <= EDA_END].copy()
L["ym"] = L.lease_date.dt.to_period("M")
print("adverts all", len(Lall), "| used for rent EDA", len(L), "| input flats", len(F))

h("1. Volume and coverage by year (counts only, all dates)")
y = Lall.assign(y=Lall.lease_date.dt.year, linked=Lall.gnaf_id.isin(F.gnaf_id))
print(y.groupby("y").agg(n=("rent", "size"), flats=("gnaf_id", "nunique"), linked=("linked", "mean"), was_leased=("was_leased", "mean"),
                         park_blank=("advert_car_space", lambda s: s.isna().mean())).round(3).to_string())
h("1b. Monthly volume, last 16 months: counts only")
m = Lall[Lall.lease_date >= "2025-04-01"].assign(ym=lambda d: d.lease_date.dt.to_period("M"), linked=lambda d: d.gnaf_id.isin(F.gnaf_id))
print(m.groupby("ym").agg(n=("rent", "size"), linked=("linked", "sum"), flats=("gnaf_id", "nunique"), leased=("was_leased", "sum")).T.to_string())
print("Two-month windows, linked adverts / unique flats / with parking known in advert / was_leased (target: 1,699 scored):")
for s, e in [("2025-08-01", "2025-09-30"), ("2025-10-01", "2025-11-30"), ("2026-06-01", "2026-07-31")]:
    w = Lall[Lall.lease_date.between(s, e) & Lall.gnaf_id.isin(F.gnaf_id)]
    print(s, len(w), w.gnaf_id.nunique(), int(w.advert_car_space.notna().sum()), int(w.was_leased.sum()))

h("2. Target distribution, adverts 2024-01..EDA_END")
R = L[L.lease_date >= "2024-01-01"]
print(R.rent.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).round(0).to_string())
print("skew rent", round(R.rent.skew(), 2), "skew log", round(np.log(R.rent).skew(), 2))
print(R.assign(b=R.bedrooms.clip(upper=4)).groupby("b").rent.describe(percentiles=[.1, .5, .9])[["count", "10%", "50%", "90%"]].round(0).to_string())
print(R.groupby("property_type").rent.agg(["size", "median"]).to_string())
g = R.dropna(subset=["car_space_resolved"]).groupby(["bedrooms", "car_space_resolved"]).rent.median().unstack().loc[0:3]
print("median rent by bed x parking\n", g.to_string())
print("garage spaces:\n", R.groupby(R.car_spaces.clip(upper=3)).rent.agg(["size", "median"]).to_string())

h("3. Noise floor A: same flat advertised twice, raw dollars")
d = L.dropna(subset=["gnaf_id"]).sort_values(["gnaf_id", "lease_date"])[["gnaf_id", "lease_date", "rent", "bedrooms", "car_space_resolved"]].copy()
d["pr"], d["pd"], d["pb"], d["pc"] = (d.groupby("gnaf_id")[c].shift() for c in ("rent", "lease_date", "bedrooms", "car_space_resolved"))
d = d.dropna(subset=["pr"]); d["gap"] = (d.lease_date - d.pd).dt.days; d["dr"] = d.rent - d.pr
rec = d[d.lease_date >= "2022-01-01"]
bins = [-1, 0, 14, 30, 60, 90, 180, 365, 730, 1825, 99999]
t = rec.groupby(pd.cut(rec.gap, bins), observed=True).agg(n=("dr", "size"), med_abs=("dr", lambda s: s.abs().median()), same=("dr", lambda s: (s == 0).mean()),
                                                        med=("dr", "median"), p90abs=("dr", lambda s: s.abs().quantile(.9)))
print(t.round(3).to_string())
print("Same attributes only, gaps 15-90d, changed rent only:",
      rec[(rec.gap.between(15, 90)) & (rec.dr != 0) & (rec.bedrooms == rec.pb)].dr.abs().median())

h("4. Noise floor B: leave-one-out peers, adverts 2024-07..EDA_END, deflated by month")
P = L[L.lease_date >= "2024-07-01"].dropna(subset=["bedrooms", "gnaf_id"]).copy()
P["ly"] = np.log(P.rent)
P["ly"] -= P.groupby("ym").ly.transform("median") - P.ly.median()     # city month effect
P["adj"] = np.exp(P.ly)
def loo(keys, minn=3):
    g = P.groupby(keys, observed=True).ly
    n, s = g.transform("size"), g.transform("sum")
    est = np.exp((s - P.ly) / (n - 1).where(n >= minn))
    e = (est - P.adj).abs(); ok = e.notna()
    return dict(level="x".join(keys), cover=round(ok.mean(), 3), med_abs=round(e[ok].median(), 1), pct=round((e[ok] / P.adj[ok]).median() * 100, 1))
P["car"] = P.car_space_resolved.fillna(-1); P["bath"] = P.bathrooms.fillna(-1)
P["mb"] = P.gnaf_id.map(F.set_index("gnaf_id").mb)
lv = [["suburb"], ["bedrooms"], ["suburb", "bedrooms"], ["suburb", "bedrooms", "car"], ["mb", "bedrooms", "car"], ["building_key"], ["building_key", "bedrooms"],
      ["building_key", "bedrooms", "car"], ["building_key", "bedrooms", "bath", "car"], ["gnaf_id"]]
print(pd.DataFrame([loo(k) for k in lv]).to_string(index=False))

h("5. Same peers: spread by building type (bldg x bed x bath x car, 3+ peers)")
g = P.groupby(["building_key", "bedrooms", "bath", "car"]).ly.agg(["size", "std"]).query("size>=5")
print("cells", len(g), "median within-cell sd of log rent", round(g["std"].median(), 3), "p90", round(g["std"].quantile(.9), 3))

h("6. Data quality")
print("rent<200", int((L.rent < 200).sum()), "rent>3000", int((L.rent > 3000).sum()), "beds>5", int((L.bedrooms > 5).sum()), "beds blank", int(L.bedrooms.isna().sum()))
k = L.dropna(subset=["gnaf_id"]).groupby(["gnaf_id", "lease_date"]).rent.agg(["size", "nunique"])
print("same flat same day rows>1:", int((k["size"] > 1).sum()), "with different rents:", int((k["nunique"] > 1).sum()))
bb = L.groupby(["building_key", "bedrooms"]).rent.transform("median"); r = np.log(L.rent / bb).abs()
print("adverts >2x or <0.5x their building x bed all-time median:", int((r > np.log(2)).sum()), f"({(r > np.log(2)).mean():.2%})")
own = d[(d.gap <= 365)]; rr = (own.rent / own.pr)
print("consecutive adverts within a year, ratio >1.5 or <0.67:", int(((rr > 1.5) | (rr < .67)).sum()), f"({((rr > 1.5) | (rr < .67)).mean():.2%})")
print("within a year, ratio near 4.33 or 1/4.33 (monthly vs weekly):", int((((rr > 3.8) & (rr < 4.8)) | ((rr < .26) & (rr > .21))).sum()))
j = L.merge(F[["gnaf_id", "bedrooms", "bathrooms"]].rename(columns=dict(bedrooms="fb", bathrooms="fbb")), on="gnaf_id")
print("advert bedrooms differs from input:", f"{(j.bedrooms != j.fb).mean():.2%}", "| since 2024:", f"{(j[j.lease_date >= '2024-01-01'].bedrooms != j[j.lease_date >= '2024-01-01'].fb).mean():.2%}")
print("was_leased rent gap since 2024, median rent by flag:", R.groupby("was_leased").rent.median().to_dict(), "share 1:", round(R.was_leased.mean(), 3))
print("input property_type:", F.property_type.value_counts().to_dict())
print("input bedrooms:", F.bedrooms.value_counts().sort_index().to_dict())

h("7. Who is scored? Hazard of re-advertising, and shift from all flats")
print("Months between consecutive adverts of a flat, since 2022 (share):")
gm = (rec.gap / 30.44).round().clip(upper=37)
print((gm.value_counts(normalize=True).sort_index() * 100).round(1).to_dict())
A = L[L.gnaf_id.isin(F.gnaf_id)]
first = A.groupby("gnaf_id").lease_date.transform("min") == A.lease_date
print("first-time share of linked adverts by half-year:")
hy = A.assign(first=first, hy=A.lease_date.dt.year.astype(str) + "H" + ((A.lease_date.dt.month > 6) + 1).astype(str))
print(hy[hy.lease_date >= "2022-01-01"].groupby("hy")["first"].mean().round(3).to_dict())
last = A.groupby("gnaf_id").lease_date.max().reindex(F.gnaf_id)
age = (EDA_END - last).dt.days.values / 365.25
ages = pd.cut(pd.Series(age).fillna(99), [-1, .5, 1, 2, 3, 5, 10, 98, 100], labels=["<6m", "6-12m", "1-2y", "2-3y", "3-5y", "5-10y", ">10y", "never"])
win = Lall[Lall.lease_date.between("2025-10-01", EDA_END)]
print("All flats by age of last advert at EDA_END:", (ages.value_counts(normalize=True).sort_index() * 100).round(1).to_dict())
c0 = pd.Timestamp("2025-09-30"); w = L[L.lease_date.between("2025-10-01", "2025-11-30") & L.gnaf_id.isin(F.gnaf_id)]
pa = L[L.lease_date <= c0].groupby("gnaf_id").lease_date.max()
wa = (c0 - w.gnaf_id.map(pa)).dt.days / 365.25
ws = pd.cut(wa.fillna(99), [-1, .5, 1, 2, 3, 5, 10, 98, 100], labels=["<6m", "6-12m", "1-2y", "2-3y", "3-5y", "5-10y", ">10y", "never"])
print("Oct-Nov 2025 adverts by age of previous advert:", (ws.value_counts(normalize=True).sort_index() * 100).round(1).to_dict())
fm = F.set_index("gnaf_id")
cmp_ = pd.DataFrame({"all_flats": [F.bedrooms.mean(), F.car_space.mean(), F.flats_in_building.median(), F.floor_level.isna().mean()],
                     "adverts": [fm.loc[w.gnaf_id].bedrooms.mean(), fm.loc[w.gnaf_id].car_space.mean(), fm.loc[w.gnaf_id].flats_in_building.median(), fm.loc[w.gnaf_id].floor_level.isna().mean()]},
                    index=["bedrooms", "parking", "flats_in_bldg", "floor_blank"]).round(3)
print(cmp_.to_string())
sa = (F.suburb.value_counts(normalize=True) * 100).round(1); sw = (w.suburb.value_counts(normalize=True) * 100).round(1)
print("largest suburb share gaps (adverts minus all):", (sw.sub(sa, fill_value=0)).sort_values().round(1).iloc[[0, 1, 2, -3, -2, -1]].to_dict())

h("8. Building data depth at EDA_END")
b12 = L[L.lease_date > EDA_END - pd.DateOffset(months=12)].groupby("building_key").size().reindex(F.building_key.unique()).fillna(0)
print("buildings", len(b12), "with 0/1-4/5-19/20+ adverts in 12m:", [int((b12 == 0).sum()), int(b12.between(1, 4).sum()), int(b12.between(5, 19).sum()), int((b12 >= 20).sum())])
cnt = F.building_key.map(b12)
print("share of flats in those buildings:", [round(float(x), 3) for x in [(cnt == 0).mean(), cnt.between(1, 4).mean(), cnt.between(5, 19).mean(), (cnt >= 20).mean()]])
