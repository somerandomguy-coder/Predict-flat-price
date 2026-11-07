"""Fill api_cache/. Safe to rerun: anything cached is skipped. Stages: suburbs, adverts, static, sales."""
import sys
from collections import Counter

from src import api
from src.load import load_input, load_leases

PAGE = 1000


def covered_suburbs() -> list:
    return api.get("me")["area"]["suburbs"]


def representatives() -> list:
    """One flat per building. 99% of keys span under about 70 m; suburb is included because a key does not name one."""
    f = load_input()
    f = f[f.suburb.isin(covered_suburbs())]
    return f.drop_duplicates(["suburb", "building_key"]).gnaf_id.astype(str).tolist()


def report(tag: str, out: list) -> None:
    kinds = Counter("denied" if isinstance(r, dict) and r.get("denied") else type(r).__name__ for r in out)
    print(tag, dict(kinds), flush=True)


def pull_suburbs() -> None:
    tails = ["market/median-rent-series", "market/vacancy-rate-series", "shapes/busy-roads", "shapes/noise-heatmap",
             "development/building-controls", "crime/by-mesh-block"]
    report("suburbs", api.get_many([(api.suburb_path(s, t), {}) for s in covered_suburbs() for t in tails], progress_every=0))


def pull_adverts() -> None:
    for s in covered_suburbs():
        path, offset = api.suburb_path(s, "listings/rent-history"), 0
        while True:
            d = api.get(path, dict(to_date=api.CUTOFF, limit=PAGE, offset=offset))["data"]
            offset += d["meta"]["returned"]
            if offset >= d["meta"]["total"] or not d["meta"]["returned"]:
                break
        print(s, d["meta"]["total"], flush=True)


def pull_static() -> None:
    reps = representatives()
    print("representatives", len(reps), flush=True)
    for tail, params in (("transport/nearby", dict(radius_m=2000)), ("risks/all", {}), ("amenities/nearby", {}),
                         ("schools/nearby", {}), ("development/all", {})):
        report(tail, api.get_many([(f"properties/{g}/{tail}", params) for g in reps]))


def pull_sales() -> None:
    """Flats with adverts in the panel's window go first: they decide the Fold B result. Final rows need the rest."""
    f = load_input()
    f = f[f.suburb.isin(covered_suburbs())]
    L = load_leases()
    busy = set(L[L.lease_date >= "2024-01-01"].gnaf_id.dropna().astype(str))
    ids = f.gnaf_id.astype(str).tolist()
    ids = [g for g in ids if g in busy] + [g for g in ids if g not in busy]
    print("flats", len(ids), "with panel adverts", sum(g in busy for g in ids), flush=True)
    report("sale-history", api.get_many([(f"properties/{g}/sale-history", {}) for g in ids], workers=2, progress_every=5000))


if __name__ == "__main__":
    stages = dict(suburbs=pull_suburbs, adverts=pull_adverts, static=pull_static, sales=pull_sales)
    for name in sys.argv[1:] or stages:
        stages[name]()
