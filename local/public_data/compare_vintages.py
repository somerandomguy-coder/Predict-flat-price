"""Builds the Feb 2026 G-NAF table and reports how it differs from May 2026. Aggregates only."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import external as E  # noqa: E402

feb = E.build_gnaf("feb2026").set_index("gnaf_id")
may = pd.read_parquet(E.DERIVED / "t_gnaf_may2026.parquet").set_index("gnaf_id")
print("flats: feb", len(feb), "may", len(may), "latest created in feb", feb.g_created.max())
j = feb.join(may, rsuffix="_m", how="inner")
for c in ("g_sp_number", "g_lot", "g_floor", "g_plan_lots"):
    same = (j[c] == j[f"{c}_m"]) | (j[c].isna() & j[f"{c}_m"].isna())
    print(c, "same", round(same.mean(), 4), "blank feb", round(j[c].isna().mean(), 4), "blank may", round(j[f"{c}_m"].isna().mean(), 4))
