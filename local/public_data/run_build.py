"""argv: repo root, then builder names. Run with python -I so downloads cannot shadow imports."""
import sys, time
sys.path.insert(0, sys.argv[1])
from src import external as E
for name in sys.argv[2:]:
    t = time.time()
    out = getattr(E, f"build_{name}")()
    print(name, out.shape, round(time.time() - t, 1), "s")
    print(out.drop(columns=[c for c in ("gnaf_id", "building_key", "mb_key") if c in out]).describe().T[["count", "mean", "50%", "max"]].round(2).to_string())
