"""Cache the panel and final rows with the raw-last-rent features."""
import sys
sys.path.insert(0, ".")
import pandas as pd
from local.zoo import lib as Z
from src import config as C
from src.load import load_leases
from src.predict import final_rows

Z.ZOO.mkdir(parents=True, exist_ok=True)
L = load_leases()
p = pd.read_parquet(C.OUT_DIR / "panel_n24.parquet")
p["origin"], p["target_date"] = pd.to_datetime(p.origin), pd.to_datetime(p.target_date)
f = final_rows().assign(origin=C.AS_OF)
f["target_date"] = pd.to_datetime(f.target_date)
Z.add_raw(p, L).to_parquet(Z.PANEL)
Z.add_raw(f, L).to_parquet(Z.FINAL)
q, g = Z.load()
print("panel", q.shape, "final", g.shape)
print("blank raw_prev_ratio: panel", round(q.raw_prev_ratio.isna().mean(), 3), "final", round(g.raw_prev_ratio.isna().mean(), 3))
print("best_anchor>0 final", (g.best_anchor > 0).mean(), "suburb dtype", q.suburb.dtype, g.suburb.dtype)
