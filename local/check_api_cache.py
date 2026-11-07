"""Prove the cache is whole and bounded: every expected response is present, parses, and holds nothing after the as-of date.

Reads the cache only. A killed pull leaves a *.tmp file or a missing response, and both are counted here.
"""
import json
from collections import Counter

from local.pull_api import PAGE, covered_suburbs, representatives
from src import api
from src.load import load_input


def check(jobs: list, tag: str) -> None:
    c = Counter()
    for path, params in jobs:
        f = api._cache_path(api._endpoint(path), path, dict(params))
        if not f.exists():
            c["missing"] += 1
            continue
        try:
            b = json.loads(f.read_text(encoding="utf-8"))
        except ValueError:
            c["corrupt"] += 1
            continue
        c["denied" if b.get("denied") else "ok"] += 1
        c["later_than_cutoff"] += api.later_dates(b, api.CUTOFF) > 0
    print(tag, len(jobs), dict(c), flush=True)


def main() -> None:
    print("tmp files left behind", sum(1 for _ in api.CACHE_DIR.rglob("*.tmp")))
    subs = covered_suburbs()
    for tail in ("market/median-rent-series", "market/vacancy-rate-series", "shapes/busy-roads", "shapes/noise-heatmap",
                 "development/building-controls", "crime/by-mesh-block"):
        check([(api.suburb_path(s, tail), {}) for s in subs], tail)
    pages = []
    for s in subs:
        path, offset = api.suburb_path(s, "listings/rent-history"), 0
        while True:
            params = dict(to_date=api.CUTOFF, limit=PAGE, offset=offset)
            pages.append((path, params))
            b = api.cached(path, params)
            if not b:
                break
            offset += b["data"]["meta"]["returned"]
            if offset >= b["data"]["meta"]["total"] or not b["data"]["meta"]["returned"]:
                break
    check(pages, "listings/rent-history pages")
    reps = representatives()
    for tail, params in (("transport/nearby", dict(radius_m=2000)), ("risks/all", {}), ("amenities/nearby", {}),
                         ("schools/nearby", {}), ("development/all", {})):
        check([(f"properties/{g}/{tail}", params) for g in reps], tail)
    f = load_input()
    check([(f"properties/{g}/sale-history", {}) for g in f[f.suburb.isin(subs)].gnaf_id.astype(str)], "sale-history")


if __name__ == "__main__":
    main()
