"""Microburbs API client. Every leak rule for API data is enforced here, before anything reaches disk.

- Only endpoints on the allowlist can be called, so a banned endpoint cannot be reached by accident.
- `listings/rent-history` must carry to_date <= AS_OF; a response holding any later advert is discarded.
- Dated series are cut to AS_OF before they are cached. Folds apply their own earlier cutoff at join time.
The key is read from .env, sent only in a header, and never stored or printed. The cache holds response bodies only.
"""
import hashlib
import json
import os
import random
import re
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from . import config as C

BASE = "https://api-ht.microburbs.com.au/v1"
PROJECT_DIR = C.DATA_DIR.parent  # worktrees have no data/ or .env of their own, so follow FLATS_DATA_DIR
CACHE_DIR = Path(os.environ.get("FLATS_API_CACHE", PROJECT_DIR / "api_cache"))
MAX_WORKERS = 8
MAX_TRIES = 6
# Cloudflare refuses the default Python-urllib signature (error 1010). This names the client honestly; it is not a browser string.
USER_AGENT = "flats-hackathon-team51/1.0 (2 Bulbs 1 Source; rent prediction)"
CUTOFF = C.AS_OF.strftime("%Y-%m-%d")

# name -> path pattern. Anything else is refused: undated or forward-looking data must not get in by accident.
ALLOWED = {
    "ping": r"ping", "me": r"me", "geocode_suburb": r"geocode/suburb",
    "rent_history": r"suburbs/[^/]+/listings/rent-history",
    "rent_series": r"suburbs/[^/]+/market/median-rent-series",
    "vacancy_series": r"suburbs/[^/]+/market/vacancy-rate-series",
    "sale_history": r"properties/[^/]+/sale-history",
    "transport": r"properties/[^/]+/transport/nearby",
    "amenities": r"properties/[^/]+/amenities/nearby",
    "schools": r"properties/[^/]+/schools/nearby",
    "risks": r"properties/[^/]+/risks/all",
    "zoning": r"properties/[^/]+/development/all",
    "profile": r"properties/[^/]+/profile",
    "busy_roads": r"suburbs/[^/]+/shapes/busy-roads",
    "noise": r"suburbs/[^/]+/shapes/noise-heatmap",
    "controls": r"suburbs/[^/]+/development/building-controls",
    "crime": r"suburbs/[^/]+/crime/by-mesh-block",
}
_ALLOWED_RE = {k: re.compile(f"^{v}$") for k, v in ALLOWED.items()}


class LeakError(Exception):
    """A response held data later than the cutoff, or a call had no date bound. Nothing was written."""


@lru_cache(maxsize=1)
def _key() -> str:
    for line in (PROJECT_DIR / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("MICROBURBS_API_KEY="):
            return line.split("=", 1)[1].strip().strip("'\"")
    raise RuntimeError("MICROBURBS_API_KEY is not in .env")


def _endpoint(path: str) -> str:
    hit = next((k for k, r in _ALLOWED_RE.items() if r.match(path)), None)
    if hit is None:
        raise PermissionError(f"endpoint is not on the allowlist: {path}")
    return hit


def cut_dated(obj, cutoff: str):
    """Drop every record with a `date` later than cutoff, at any depth. ISO dates compare correctly as text."""
    if isinstance(obj, list):
        return [cut_dated(x, cutoff) for x in obj if not (isinstance(x, dict) and str(x.get("date", ""))[:10] > cutoff)]
    if isinstance(obj, dict):
        return {k: cut_dated(v, cutoff) for k, v in obj.items()}
    return obj


def later_dates(obj, cutoff: str) -> int:
    """Count `date` or `listing_date` values after cutoff, at any depth."""
    if isinstance(obj, list):
        return sum(later_dates(x, cutoff) for x in obj)
    if isinstance(obj, dict):
        n = 0
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                n += later_dates(v, cutoff)
            elif k in ("date", "listing_date") and str(v)[:10] > cutoff:
                n += 1
        return n
    return 0


def _clean(name: str, body: dict) -> dict:
    """Apply the leak rule for this endpoint. Raises rather than returns if a bound failed."""
    if name == "rent_history":
        rows = body["data"]["results"]
        if later_dates(body, CUTOFF) or any(not r.get("listing_date") for r in rows):
            raise LeakError("rent-history returned a later or undated advert; response discarded")
    elif name in ("rent_series", "vacancy_series", "sale_history"):
        body = cut_dated(body, CUTOFF)
        if isinstance(body.get("data"), dict):
            body["data"].pop("as_of", None)  # says when the series was built, which is not information we had
        if later_dates(body, CUTOFF):
            raise LeakError(f"{name} still holds later dates after filtering; response discarded")
    return body


def _cache_path(name: str, path: str, params: dict) -> Path:
    tag = hashlib.sha1(json.dumps([path, sorted(params.items())]).encode()).hexdigest()[:20]
    return CACHE_DIR / name / f"{tag}.json"


_local = threading.local()


def _fetch(path: str, params: dict) -> dict | None:
    """One GET with backoff on 429 and 5xx. Returns None for the API's own JSON 403 (outside coverage)."""
    url = f"{BASE}/{path}" + (f"?{urllib.parse.urlencode(params)}" if params else "")
    if not hasattr(_local, "ctx"):
        _local.ctx = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE"))
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {_key()}", "Accept": "application/json", "User-Agent": USER_AGENT})
    for attempt in range(MAX_TRIES):
        wait = 0.0
        try:
            with urllib.request.urlopen(req, timeout=90, context=_local.ctx) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 403:
                text = e.read(2000).decode("utf-8", "replace")
                if text.lstrip().startswith("{"):
                    return None
                raise RuntimeError(f"403 from the edge, not the API, on {path}: {text[:80]}") from None
            if e.code != 429 and e.code < 500:
                raise RuntimeError(f"HTTP {e.code} on {path}") from None
            wait = float(e.headers.get("Retry-After") or 0)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(max(wait, 2 ** attempt + random.random()))
    raise RuntimeError(f"gave up after {MAX_TRIES} tries on {path}")


def get(path: str, params: dict | None = None, refresh: bool = False):
    """Cached GET. A 403 is cached too, as {"denied": true}, so uncovered suburbs are not retried."""
    params = dict(params or {})
    name = _endpoint(path)
    if name == "rent_history" and ("to_date" not in params or str(params["to_date"]) > CUTOFF):
        raise LeakError("rent-history needs to_date on or before the cutoff")
    target = _cache_path(name, path, params)
    if target.exists() and not refresh:
        return json.loads(target.read_text(encoding="utf-8"))
    body = _fetch(path, params)
    body = {"denied": True} if body is None else _clean(name, body)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(f".{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(body), encoding="utf-8")
    tmp.replace(target)
    return body


def get_many(jobs: list, workers: int = MAX_WORKERS, progress_every: int = 2000) -> list:
    """Run (path, params) jobs, at most MAX_WORKERS at a time. Failures come back as the exception, not raised."""
    done = [0]
    started = time.time()

    def one(job):
        try:
            return get(*job)
        except (RuntimeError, OSError, ValueError) as e:
            return e
        finally:
            done[0] += 1
            if progress_every and done[0] % progress_every == 0:
                print(f"{done[0]}/{len(jobs)} in {time.time() - started:.0f}s", flush=True)

    with ThreadPoolExecutor(min(workers, MAX_WORKERS)) as pool:
        return list(pool.map(one, jobs))


def suburb_path(suburb: str, tail: str) -> str:
    return f"suburbs/{urllib.parse.quote(suburb, safe='')}/{tail}"
