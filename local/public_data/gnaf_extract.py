"""Stream one G-NAF zip, keep NSW postcodes 2000-2050 (public filter), and write a parquet. No extraction to disk."""
import sys, zipfile
from pathlib import Path
import pandas as pd

zip_path, out = sys.argv[1], Path(sys.argv[2])
z = zipfile.ZipFile(zip_path)
names = z.namelist()
def member(end):
    return [n for n in names if n.endswith(end) and "/Standard/" in n][0]

cols = ["ADDRESS_DETAIL_PID", "DATE_CREATED", "DATE_LAST_MODIFIED", "DATE_RETIRED", "BUILDING_NAME", "FLAT_TYPE_CODE",
        "FLAT_NUMBER_PREFIX", "FLAT_NUMBER", "FLAT_NUMBER_SUFFIX", "LEVEL_TYPE_CODE", "LEVEL_NUMBER_PREFIX", "LEVEL_NUMBER",
        "LEVEL_NUMBER_SUFFIX", "NUMBER_FIRST", "NUMBER_LAST", "STREET_LOCALITY_PID", "POSTCODE", "ALIAS_PRINCIPAL",
        "LEGAL_PARCEL_ID", "CONFIDENCE", "ADDRESS_SITE_PID", "PRIMARY_SECONDARY", "GNAF_PROPERTY_PID"]
parts = []
with z.open(member("NSW_ADDRESS_DETAIL_psv.psv")) as f:
    for c in pd.read_csv(f, sep="|", usecols=cols, dtype=str, chunksize=500_000, keep_default_na=False, na_values=[""]):
        pc = pd.to_numeric(c.POSTCODE, errors="coerce")
        parts.append(c[pc.between(2000, 2050)])
d = pd.concat(parts, ignore_index=True)
print("rows", len(d), flush=True)

keep = set(d.ADDRESS_DETAIL_PID)
mb = []
with z.open(member("NSW_ADDRESS_MESH_BLOCK_2021_psv.psv")) as f:
    for c in pd.read_csv(f, sep="|", usecols=["ADDRESS_DETAIL_PID", "MB_2021_PID", "DATE_RETIRED"], dtype=str, chunksize=1_000_000):
        mb.append(c[c.ADDRESS_DETAIL_PID.isin(keep) & c.DATE_RETIRED.isna()].drop(columns="DATE_RETIRED"))
mb = pd.concat(mb).drop_duplicates("ADDRESS_DETAIL_PID")
with z.open(member("NSW_MB_2021_psv.psv")) as f:
    code = pd.read_csv(f, sep="|", usecols=["MB_2021_PID", "MB_2021_CODE"], dtype=str)
mb = mb.merge(code, on="MB_2021_PID", how="left").drop(columns="MB_2021_PID")
d = d.merge(mb, on="ADDRESS_DETAIL_PID", how="left")

geo = []
with z.open(member("NSW_ADDRESS_DEFAULT_GEOCODE_psv.psv")) as f:
    for c in pd.read_csv(f, sep="|", usecols=["ADDRESS_DETAIL_PID", "GEOCODE_TYPE_CODE", "LONGITUDE", "LATITUDE"], dtype=str, chunksize=1_000_000):
        geo.append(c[c.ADDRESS_DETAIL_PID.isin(keep)])
d = d.merge(pd.concat(geo).drop_duplicates("ADDRESS_DETAIL_PID"), on="ADDRESS_DETAIL_PID", how="left")
out.parent.mkdir(parents=True, exist_ok=True)
d.to_parquet(out)
print(d.shape, d.DATE_CREATED.min(), d.DATE_CREATED.max(), d.DATE_LAST_MODIFIED.max())
