"""Coverage per source over all input flats, and how far each one fills the supplied gaps. Aggregates only."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config as C  # noqa: E402
from src import external as E  # noqa: E402
from src.load import load_input  # noqa: E402

f = E.attach(load_input(), cutoff=C.AS_OF)
print("flats", len(f), "buildings", f.building_key.nunique())
for group, cols in E.GROUPS.items():
    for c in cols:
        if c in f:
            print(f"{group:13s} {c:22s} filled {f[c].notna().mean():6.1%}")
b = f.floor_level.isna()
print("floor blank", int(b.sum()), "filled by g_floor", int(f.g_floor[b].notna().sum()))
print("floor blank with lot rank", int(f.g_lot_rank[b].notna().sum()), "with flat number", int(f.g_flat_num[b].notna().sum()))
k = f.block_homes_2026.isna()
print("block blank", int(k.sum()), "with census dwellings", int(f.mb_dwellings_2021[k].notna().sum()), "with census rent", int(f.c_median_rent[k].notna().sum()))
print("block-blank census dwellings: median", f.mb_dwellings_2021[k].median(), "zero share", round((f.mb_dwellings_2021[k] == 0).mean(), 3))
p = f.car_space.isna()
print("parking blank", int(p.sum()), "with strata plan", int(f.g_sp_number[p].notna().sum()))
print("share of flats in buildings with OSM levels", round(f.osm_levels.notna().mean(), 3), "with footprint", round(f.osm_footprint_m2.notna().mean(), 3))
