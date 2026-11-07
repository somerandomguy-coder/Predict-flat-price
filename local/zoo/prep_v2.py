import sys
sys.path.insert(0, ".")
from local.zoo import lib as Z
from src.load import load_leases
L = load_leases()
p, f = Z.load()
for d, path in ((p, Z.PANEL), (f, Z.FINAL)):
    n = len(d); o = d.gnaf_id.astype(str).to_numpy()
    x = Z.add_v2(d, L)
    assert len(x) == n and (x.gnaf_id.astype(str).to_numpy() == o).all()
    t = path.with_suffix(".tmp"); x.to_parquet(t); t.replace(path)
    print(path.name, x.shape, x[Z.V2_COLS].isna().mean().round(3).to_dict())
