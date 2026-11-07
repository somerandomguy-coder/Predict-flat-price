"""Real-data leak audit. Uses Fold B for ablations, so Fold A is not read for any choice."""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

from src import config as C
from src import features as F
from src import models as M
from src.load import fold_test_rows, load_input, load_leases
from src.predict import final_rows

out = {}

# 1. Files: is anything dated after the as-of date?
raw = pd.read_csv(C.LEASES_CSV, usecols=["listing_date"]).listing_date
inp = load_input()
out["files"] = dict(lease_max=str(raw.max()), attrs_max=str(inp.attrs_last_seen.max().date()), listed_max=str(inp.first_ever_listed.max().date()))

# 2. Poison every row after the cutoff in the real leases. No feature may move.
cut = C.FOLDS["B"]["cutoff"]
t = fold_test_rows("B")
base = F.build_features(cut, t)
L = load_leases().copy()
late = L.lease_date > cut
L.loc[late, "rent"] *= 5
L.loc[late, "bedrooms"] = 9
L.loc[late, "advert_car_space"] = 1 - L.loc[late, "advert_car_space"]
orig = F.load_leases
F.load_leases = lambda: L
pois = F.build_features(cut, t)
F.load_leases = orig
cols = [c for c in F.FEATURE_COLS if c in base.columns]
moved = [c for c in cols if not base[c].astype(object).where(base[c].notna(), None).equals(pois[c].astype(object).where(pois[c].notna(), None))]
out["poison_moved_features"] = moved

# 3. Do fold rows look like the rows we actually predict? NaN-rate skew, then a classifier.
panel = pd.read_parquet(C.OUT_DIR / "features_panel.parquet")
fb = M.fold_rows(panel, "B")
fin = final_rows()
skew = pd.DataFrame({"fold_nan": fb[cols].isna().mean(), "final_nan": fin[cols].isna().mean()})
skew["gap"] = (skew.fold_nan - skew.final_nan).abs()
out["nan_skew"] = skew.sort_values("gap", ascending=False).head(8).round(3).to_dict("index")
num = [c for c in cols if c not in ("suburb", "property_type")]
A = pd.concat([fb[num].assign(y=1), fin[num].sample(len(fb) * 3, random_state=1).assign(y=0)]).astype(float)
auc = {}
for name, use in (("all", num), ("no_date_ages", [c for c in num if c not in ("attrs_age_days", "listed_age_days", "attrs_missing", "listed_missing")])):
    auc[name] = float(cross_val_score(HistGradientBoostingClassifier(max_iter=100), A[use], A.y, cv=3, scoring="roc_auc").mean())
out["fold_vs_final_auc"] = auc

# 4. Ablate suspects on Fold B.
sets = {"all_features": [], "no_attrs_age": ["attrs_age_days"],
        "no_date_ages": ["attrs_age_days", "listed_age_days", "attrs_missing", "listed_missing"],
        "no_date_ages_no_ever": ["attrs_age_days", "listed_age_days", "attrs_missing", "listed_missing", "ever_advertised_for_lease"]}
out["ablation_fold_b"] = {}
for name, drop in sets.items():
    r = M.run_experiment(dict(name=f"leak_{name}", frame=panel, methods=["direct"], drop_features=drop,
                              description=f"Leak audit Fold B ablation: {name}"))
    out["ablation_fold_b"][name] = {k: round(v, 3) for k, v in r["folds"]["B"].items() if isinstance(v, float)}

# 5. First-time flats: their input fields may come from the test advert itself. How much does that matter?
model = M.fit_model(M.train_rows(panel, cut, 12), [c for c in cols], "direct", {})
first = fb.anchor_own.isna().to_numpy()
rng = np.random.default_rng(3)
noisy = fb.copy()
hit = first & (rng.random(len(fb)) < 0.05)
noisy.loc[hit, "bedrooms"] = noisy.loc[hit, "bedrooms"] + rng.choice([-1, 1], hit.sum())
for name, d in (("clean", fb), ("bedrooms_5pct_wrong", noisy)):
    r = M.evaluate(model, d[first])
    out.setdefault("first_time_flats", {})[name] = {k: round(v, 3) for k, v in r.items() if isinstance(v, (int, float))}
r = M.evaluate(model, fb[~first])
out["repeat_flats"] = {k: round(v, 3) for k, v in r.items() if isinstance(v, (int, float))}

(C.OUT_DIR / "leak_audit.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out, indent=1, default=str))
