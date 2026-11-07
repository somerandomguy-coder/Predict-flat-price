"""Fill api_cache/. Safe to rerun: anything cached is skipped. Stages: suburbs, adverts, static, sales."""
import sys
from collections import Counter

from src import api
from src.load import load_input

PAGE = 1000


def covered_suburbs() -> list:
    return api.get("me")["area"]["suburbs"]


def representatives() -> list:
    """One flat per unique coordinate in a covered suburb. A building_key can span several coordinates."""
    f = load_input()
    f = f[f.suburb.isin(covered_suburbs())]
    return f.drop_duplicates(["latitude", "longitude"]).gnaf_id.astype(str).tolist()


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
    f = load_input()
    ids = f[f.suburb.isin(covered_suburbs())].gnaf_id.astype(str).tolist()
    report("sale-history", api.get_many([(f"properties/{g}/sale-history", {}) for g in ids], progress_every=5000))


if __name__ == "__main__":
    stages = dict(suburbs=pull_suburbs, adverts=pull_adverts, static=pull_static, sales=pull_sales)
    for name in sys.argv[1:] or stages:
        stages[name]()
