"""Same blend and premium as the main file, without the $10 rounding. Writes under out/ only and reads no fold."""
import sys, json
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from local.zoo import lib as Z, stack as S
from src import config as C, models as M, parking
from src.validate import check_submission, round_half
from src.load import load_input
w = json.loads((Z.ZOO / "written.json").read_text()); wts = w["weights"]
_, X = Z.load()
F = {n: pd.read_parquet(Z.PRED / n / "F.parquet") for n in wts}
lw = sum(a * np.log(F[n].pred_with.clip(lower=M.PRED_FLOOR).to_numpy()) for n, a in wts.items())
lwo = sum(a * np.log(F[n].pred_without.clip(lower=M.PRED_FLOOR).to_numpy()) for n, a in wts.items())
prem = parking.premium_for(X, parking.estimate_premium(C.AS_OF)) if w["premium_source"] == "within_building" else None
c = S.prices(lw, lwo, X, prem)
sub = pd.DataFrame({"gnaf_id": X.gnaf_id.astype(object).to_numpy(), "rent_with_car_space": round_half(c.rent_with_car_space), "rent_without_car_space": round_half(c.rent_without_car_space)})
sub["rent_with_car_space"] = sub[["rent_with_car_space", "rent_without_car_space"]].max(axis=1)
ids = load_input().gnaf_id
sub = pd.DataFrame({"gnaf_id": ids.astype(object).to_numpy()}).merge(sub, on="gnaf_id", how="left")
check_submission(sub, ids.astype(object))
out = Z.ZOO / "rents_2-bulbs-1-source_unrounded.csv"
sub.to_csv(out, index=False); print("wrote", out, sub.shape)
a = pd.read_csv(C.SUBMISSION_CSV)
m = a.merge(sub, on="gnaf_id", suffixes=("_r", "_u"))
print("max abs diff, rounded vs unrounded, without column:", (m.rent_without_car_space_r - m.rent_without_car_space_u).abs().max(), "| median gap unrounded", (sub.rent_with_car_space - sub.rent_without_car_space).median())
