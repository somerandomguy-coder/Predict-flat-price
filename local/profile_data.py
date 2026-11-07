"""Stage 1: profile the files from raw columns, assuming nothing about the lease schema."""
import sys

import pandas as pd

from src import config as C
from src.load import load_input, load_leases


def table(name, path):
    df = pd.read_csv(path, low_memory=False)
    lines = [f"## {path.name}", f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns", "",
             "| column | dtype | nulls | null % | distinct | min | max |", "|---|---|---|---|---|---|---|"]
    for c in df.columns:
        s = df[c]
        looks_date = "date" in c.lower() or "seen" in c.lower() or "listed" in c.lower()
        show = pd.api.types.is_numeric_dtype(s) or looks_date  # never print addresses or ids
        lo, hi = (s.dropna().min(), s.dropna().max()) if show and s.notna().any() else ("", "")
        lines.append(f"| {c} | {s.dtype} | {s.isna().sum():,} | {s.isna().mean():.1%} | {s.nunique():,} | {lo} | {hi} |")
    return lines + [""]


def main():
    md = [f"# Data profile{' (MOCK DATA, NOT REAL)' if C.MOCK else ''}", ""]
    for p in (C.INPUT_CSV, C.LEASES_CSV, C.BLOCKS_CSV):
        md += table(p.stem, p)
    L, F = load_leases(), load_input()
    raw_n = sum(1 for _ in open(C.LEASES_CSV, encoding="utf-8")) - 1
    md += ["## Links and coverage", f"- Lease rows kept after as-of and rent filters: {len(L):,} of {raw_n:,}",
           f"- Lease dates kept: {L.lease_date.min():%Y-%m-%d} to {L.lease_date.max():%Y-%m-%d}",
           f"- Leases linked to an input flat: {L.gnaf_id.isin(F.gnaf_id).mean():.1%}",
           f"- Input flats with a previous lease: {F.gnaf_id.isin(L.gnaf_id).sum():,} of {len(F):,}",
           f"- Input flats whose building has a lease: {F.building_key.isin(L.building_key).sum():,}",
           f"- Advert parking blank: {L.advert_car_space.isna().mean():.1%}; after input fill: {L.car_space_resolved.isna().mean():.1%}",
           f"- Input attrs_last_seen after as-of: {(F.attrs_last_seen > C.AS_OF).sum():,}",
           f"- Input first_ever_listed after as-of: {(F.first_ever_listed > C.AS_OF).sum():,}", "", "## Fold test sizes"]
    for k, f in C.FOLDS.items():
        t = L[L.lease_date.between(f["start"], f["end"]) & L.gnaf_id.isin(F.gnaf_id)]
        md.append(f"- Fold {k}: {len(t):,} adverts, {t.car_space_resolved.notna().sum():,} scorable")
    C.PROFILE_MD.parent.mkdir(parents=True, exist_ok=True)
    C.PROFILE_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    sys.exit(main())
