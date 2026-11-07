"""Build report.html from aggregate JSON, and fill the pitch script from final numbers.

    python -m src.report            real: out/real/*.json -> report.html  (FLATS_MOCK=1: out/mock/report.html)
    python -m src.report --fixture  layout test with fake numbers, FLATS_MOCK=1 only -> out/mock/report_fixture.html
    python -m src.report --pitch    PITCH_SCRIPT.md -> PITCH_SCRIPT.filled.md, real inputs only

Safety rules:
- A value that is missing or not a finite number shows as "pending". Nothing here estimates a number.
- A real run stops, writing nothing, unless every input present is stamped {"provenance": {"mode": "real"}}.
  Lanes get the stamp from provenance() below. Missing inputs are allowed and show as pending.
- Only a real run writes ROOT/report.html. The fixture lives in this file and is never written as an input.
"""
import argparse
import html
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config as C

# ---------------------------------------------------------------- expected inputs: the one definition
# Every file sits in C.OUT_DIR, except report_text, which is hand-written (see text_path).
# make_fixture() below is a full worked example of each shape. Money is $/week. Bias is prediction minus advert,
# so negative means low. Fold metrics use the keys of src.validate.score: n, mae (MEDIAN abs error), within100
# (a 0-1 share), r2, bias. Any field may be absent; the report then says pending for that field alone.
INPUTS = {
    "metrics": dict(files=["metrics.json"], owner="model lane", shape="""
        {"provenance": {...}, "method_summary": "one plain sentence naming the final model",
         "sources": {"nsw_bond_used": false},
         "methods": [{"key": "suburb_median" | "baseline" | "final" | <any other>, "label": "Shown name",
                      "folds": {"A": {"n", "mae", "within100", "r2", "bias"}, "B": {...}}}]}"""),
    "index_bias": dict(files=["index_bias.json"], owner="features lane", shape="""
        {"provenance": {...}, "fold": "A", "overall": {"bias_before": -1, "bias_after": -1},
         "by_age": [{"age_band": "0 to 3 months", "n": 1, "bias_before": -1, "bias_after": -1}]}"""),
    "parking": dict(files=["parking.json"], owner="parking lane", shape="""
        {"provenance": {...}, "overall": {"premium": 1, "low": 1, "high": 1, "n": 1},
         "by_suburb": [{"suburb": "Name", "premium": 1, "low": 1, "high": 1, "n": 1}],
         "by_bedrooms": [{"bedrooms": 1, "premium": 1, "low": 1, "high": 1, "n": 1}]}"""),
    "floor": dict(files=["floor.json"], owner="parking lane", shape="""
        {"provenance": {...}, "per_floor_premium": 1, "reference_floor": 1,
         "curve": [{"floor": 1, "premium": 0, "low": 0, "high": 0, "n": 1}]}"""),
    "shap": dict(files=["shap.json", "shap_summary.json"], owner="model lane", shape="""
        {"provenance": {...}, "unit": "$/week",
         "top_features": [{"feature": "column_name", "label": "Plain name", "mean_abs_shap": 1}]}"""),
    "report_text": dict(files=[], owner="team, by hand", shape="""
        {"provenance": {...}, "surprising_finding": "text", "what_worked": ["text"], "what_didnt": ["text"],
         "journey": [{"step", "change", "foldA", "within100", "foldB"}]}"""),
}

# Published figures, quoted as supplied. These are the only result numbers written into this file.
MICROBURBS_MAE = 86
PUBLISHED = [  # method, median error, within $100, typical bias
    ("Simple", "$100", "not published", "not published"),
    ("Simple + parking", "$93", "not published", "not published"),
    ("Microburbs valuation (their figure)", "$86", "55.21%", "$61 low"),
    ("Suburb median", "not published", "56.39%", "not published"),
]
NSW_ATTRIBUTION = "© State of New South Wales. For current information go to www.nsw.gov.au."
MAX_BARS = 15


class Refused(Exception):
    """A safety rule stopped the run. Nothing was written."""


def provenance() -> dict:
    """Lanes put this under the "provenance" key of each JSON file they write."""
    return {"mode": "mock" if C.MOCK else "real", "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def text_path() -> Path:
    """Committed for real runs. Mock text stays under out/mock, so it cannot reach the real report."""
    return C.OUT_DIR / "report_text.json" if C.MOCK else C.ROOT / "notes" / "report_text.json"


def load_inputs() -> tuple[dict, dict]:
    """Return ({name: dict}, {name: reason}) for what loaded and what did not. Never raises on bad input."""
    data, problems = {}, {}
    for name, spec in INPUTS.items():
        paths = [C.OUT_DIR / f for f in spec["files"]] or [text_path()]
        path = next((p for p in paths if p.exists()), None)
        if path is None:
            problems[name] = f"{paths[0].name} not found yet ({spec['owner']})"
            continue
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("top level is not an object")
            data[name] = loaded
        except (ValueError, OSError) as e:
            problems[name] = f"{path.name} could not be read ({type(e).__name__})"
    return data, problems


def require_real(data: dict) -> None:
    """Stop unless this is a real environment and every loaded input says it is real."""
    if C.MOCK:
        raise Refused("FLATS_MOCK=1 is set, so this is a mock run")
    if "mock" in Path(C.DATA_DIR).parts or Path(C.MOCK_TRUTH_CSV).exists():
        raise Refused(f"data folder {C.DATA_DIR} looks like mock data")
    bad = []
    for name, d_ in data.items():
        prov = d_.get("provenance")
        mode = prov.get("mode") if isinstance(prov, dict) else None
        if mode != "real" or d_.get("fixture"):
            bad.append(f"{name} (mode={mode!r})")
    if bad:
        raise Refused("inputs are not stamped real: " + ", ".join(bad))


# ---------------------------------------------------------------- tolerant reading
def num(x):
    """A finite number, or None. Booleans and numeric strings do not count."""
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        return None
    return float(x)


def dig(d_, *path):
    for k in path:
        d_ = d_.get(k) if isinstance(d_, dict) else None
    return d_


def rows_of(x) -> list:
    return [r for r in x if isinstance(r, dict)] if isinstance(x, list) else []


def text_of(x):
    return x.strip() if isinstance(x, str) and x.strip() else None


def method(metrics, key) -> dict:
    return next((m for m in rows_of(dig(metrics, "methods")) if m.get("key") == key), {})


def fold_metrics(m, fold) -> dict:
    """One fold's metrics with field aliases and percent-style shares normalised."""
    f = dig(m, "folds", fold)
    f = f if isinstance(f, dict) else {}

    def pick(*keys):
        return next((num(f[k]) for k in keys if num(f.get(k)) is not None), None)

    share = pick("within100", "within_100")
    return dict(n=pick("n"), mae=pick("mae", "median_abs_error"), r2=pick("r2", "R2"), bias=pick("bias"),
                within100=share / 100 if share is not None and share > 1 else share)


# ---------------------------------------------------------------- formatting
PENDING = "pending"


def esc(x) -> str:
    return html.escape(str(x), quote=True)


def money(x, signed=False) -> str:
    if x is None:
        return PENDING
    a = abs(x)
    body = f"{a:,.0f}" if abs(a - round(a)) < 0.005 else f"{a:,.2f}"
    sign = "-" if x < 0 else ("+" if signed and x > 0 else "")
    return f"{sign}${body}"


def pct(x) -> str:
    return PENDING if x is None else f"{x * 100:.2f}%"


def plain(x, digits=3) -> str:
    return PENDING if x is None else f"{x:,.{digits}f}"


def count(x) -> str:
    return PENDING if x is None else f"{x:,.0f}"


def span_text(low, high) -> str:
    return PENDING if low is None or high is None else f"{money(low)} to {money(high)}"


def pending_note(what: str) -> str:
    return f'<p class="pending"><strong>Pending:</strong> {esc(what)}</p>'


def table(headers, rows, caption=None) -> str:
    head = "".join(f'<th scope="col">{esc(h)}</th>' for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in rows)
    cap = f"<caption>{esc(caption)}</caption>" if caption else ""
    return f'<div class="tablewrap"><table>{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def data_view(headers, rows) -> str:
    return f"<details><summary>Show the numbers</summary>{table(headers, rows)}</details>"


# ---------------------------------------------------------------- charts: HTML bars and inline SVG, no scripts
def legend(items) -> str:
    return '<div class="legend">' + "".join(f'<span><i class="key {c}"></i>{esc(t)}</span>' for c, t in items) + "</div>"


def hbars(rows, fmt=money) -> str:
    """rows: [(label, [(value or None, css_class, series_name)])]. Zero is a shared baseline; None draws no bar."""
    vals = [v for _, bars in rows for v, _, _ in bars if v is not None]
    lo, hi = min(vals + [0.0]), max(vals + [0.0])
    span = (hi - lo) or 1.0
    zero = (0 - lo) / span * 100
    out = []
    for label, bars in rows:
        lines = []
        for v, cls, name in bars:
            tip = f"{label}, {name}: {fmt(v)}" if name else f"{label}: {fmt(v)}"
            mark = ""
            if v is not None:
                w = abs(v) / span * 100
                left = zero if v >= 0 else zero - w
                side = "pos" if v >= 0 else "neg"
                mark = f'<span class="bar {cls} {side}" style="left:{left:.2f}%;width:{w:.2f}%"></span>'
            state = " is-pending" if v is None else ""
            lines.append(f'<div class="line" title="{esc(tip)}"><div class="track" style="--zero:{zero:.2f}%">{mark}</div>'
                         f'<div class="val{state}">{esc(fmt(v))}</div></div>')
        out.append(f'<div class="barrow"><div class="lab">{esc(label)}</div><div>{"".join(lines)}</div></div>')
    return f'<div class="bars">{"".join(out)}</div>'


def nice_ticks(lo, hi, n=5) -> list:
    raw = (hi - lo) / max(n - 1, 1) or 1.0
    mag = 10 ** math.floor(math.log10(raw))
    step = next(s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw)
    t, out = math.floor(lo / step) * step, []
    while t < hi + step * 0.999:
        out.append(t)
        t += step
    return out


def floor_chart(curve) -> str:
    """Line of premium against floor, with the uncertainty band when low and high are supplied."""
    W, H, L, R, T, B = 400, 250, 50, 12, 12, 42
    pts = sorted(curve, key=lambda p: p[0])
    band = [p for p in pts if p[2] is not None and p[3] is not None]
    ys = [p[1] for p in pts] + [p[2] for p in band] + [p[3] for p in band] + [0.0]
    ticks = nice_ticks(min(ys), max(ys))
    y0, y1 = ticks[0], ticks[-1]
    x0, x1 = pts[0][0], pts[-1][0]

    def X(v):
        return L + (v - x0) / ((x1 - x0) or 1) * (W - L - R)

    def Y(v):
        return H - B - (v - y0) / ((y1 - y0) or 1) * (H - T - B)

    s = [f'<svg class="plot" viewBox="0 0 {W} {H}" role="img" aria-label="Rent premium in dollars a week by floor">']
    for t in ticks:
        cls = "zero" if t == 0 else "grid"
        s.append(f'<line class="{cls}" x1="{L}" x2="{W - R}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>')
        s.append(f'<text class="tick" x="{L - 6}" y="{Y(t) + 4:.1f}" text-anchor="end">{esc(money(t))}</text>')
    for v in nice_ticks(x0, x1, 6):
        if x0 <= v <= x1:
            s.append(f'<text class="tick" x="{X(v):.1f}" y="{H - B + 16}" text-anchor="middle">{v:.0f}</text>')
    s.append(f'<text class="tick" x="{(L + W - R) / 2}" y="{H - 6}" text-anchor="middle">Floor</text>')
    if len(band) > 1:
        ring = [(X(p[0]), Y(p[3])) for p in band] + [(X(p[0]), Y(p[2])) for p in reversed(band)]
        s.append('<polygon class="band" points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in ring) + '"/>')
    s.append('<polyline class="curve" points="' + " ".join(f"{X(p[0]):.1f},{Y(p[1]):.1f}" for p in pts) + '"/>')
    for p in pts:
        s.append(f'<circle class="dot" cx="{X(p[0]):.1f}" cy="{Y(p[1]):.1f}" r="4">'
                 f'<title>Floor {p[0]:.0f}: {esc(money(p[1], True))} a week</title></circle>')
    return "".join(s) + "</svg>"


def d(ts, f="%-d %b %Y") -> str:
    return pd.Timestamp(ts).strftime(f)


def split_diagram() -> str:
    """Timeline drawn from src.config, so it cannot drift from the dates actually used."""
    W, L, R = 400, 8, 8
    start, end = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-10-01")

    def X(t):
        return L + (pd.Timestamp(t) - start).days / (end - start).days * (W - L - R)

    A, B = C.FOLDS["A"], C.FOLDS["B"]
    rows = [("Fold B: tuning check", B["cutoff"], B["start"], B["end"]),
            ("Fold A: held-out test", A["cutoff"], A["start"], A["end"]),
            ("Final: the submission", C.AS_OF, C.SCORE_START, C.SCORE_END)]
    H = 34 + 56 * len(rows)
    s = [f'<svg class="plot" viewBox="0 0 {W} {H}" role="img" aria-label="Date split: each fold learns from the past '
         f'and is tested on the next two months">']
    for m in pd.date_range(start, end, freq="MS")[:-1]:
        s.append(f'<line class="grid" x1="{X(m):.1f}" x2="{X(m):.1f}" y1="20" y2="{H}"/>')
        s.append(f'<text class="tick" x="{X(m) + 3:.1f}" y="14">{m.strftime("%b")}</text>')
    for i, (name, cut, a, b) in enumerate(rows):
        y = 28 + 56 * i
        s.append(f'<text class="rowlab" x="{L}" y="{y + 12}">{esc(name)}</text>')
        s.append(f'<rect class="past" x="{L}" y="{y + 20}" width="{X(cut) - L - 1:.1f}" height="20" rx="4">'
                 f'<title>Learns from data up to {d(cut)}</title></rect>')
        s.append(f'<rect class="held" x="{X(a) + 1:.1f}" y="{y + 20}" width="{X(b) - X(a) - 1:.1f}" height="20" rx="4">'
                 f'<title>{esc(name)}: {d(a)} to {d(b)}</title></rect>')
        s.append(f'<text class="inbar" x="{X(cut) - 6:.1f}" y="{y + 34}" text-anchor="end">to {d(cut, "%-d %b")}</text>')
        s.append(f'<text class="inbar dark" x="{(X(a) + X(b)) / 2:.1f}" y="{y + 34}" text-anchor="middle">'
                 f'{d(a, "%b")} to {d(b, "%b")}</text>')
    return "".join(s) + "</svg>"


# ---------------------------------------------------------------- sections
def section(n, title, lead, body) -> str:
    prefix = f"{n}. " if n else ""
    p = f'<p class="lead">{esc(lead)}</p>' if lead else ""
    return f"<section><h2>{esc(prefix + title)}</h2>{p}{body}</section>"


def tiles(data) -> str:
    f = fold_metrics(method(data.get("metrics"), "final"), "A")
    items = [(money(f["mae"]), "typical miss on held-back adverts, per week"),
             (pct(f["within100"]), "of held-back adverts within $100 a week"),
             (plain(f["r2"]), "R2 on held-out Fold A adverts")]
    cells = "".join(f'<div class="tile{" is-pending" if v == PENDING else ""}"><b>{esc(v)}</b><span>{esc(t)}</span></div>'
                    for v, t in items)
    return f'<div class="tiles">{cells}</div>'


def sec_journey(data, problems) -> str:
    """The path from baseline to now. Steps come from report_text.json, so a new step is one more entry."""
    raw = dig(data.get("report_text"), "journey")
    steps = [j for j in raw if isinstance(j, dict) and text_of(j.get("step")) and num(j.get("foldA")) is not None] \
        if isinstance(raw, list) else []
    if not steps:
        return section(None, "From $100 to now", None, pending_note("journey steps. Add journey to " + text_path().name))
    first, last = num(steps[0]["foldA"]), num(steps[-1]["foldA"])
    lead = (f"We started with a typical miss of {money(first)} a week and reached {money(last)}. "
            "Each step is scored on the same held-out date windows, as it stood at the time.")
    bars = hbars([(f"{i}. {text_of(j['step'])}", [(num(j["foldA"]), "s1", "")]) for i, j in enumerate(steps, 1)])
    rows, notes, prev = [], [], None
    for i, j in enumerate(steps, 1):
        a = num(j["foldA"])
        delta = "start" if prev is None else f"{'+' if a > prev else '-'}${abs(a - prev):.2f}"
        rows.append((f"{i}. {text_of(j['step'])}", money(a), delta, pct(num(j.get("within100"))), money(num(j.get("foldB")))))
        notes.append(f"<li><strong>{esc(text_of(j['step']))}.</strong> {esc(text_of(j.get('change')) or '')}</li>")
        prev = a
    body = bars + "<ol>" + "".join(notes) + "</ol>" + table(["Step", "Fold A", "Change", "Within $100", "Fold B"], rows)
    body += '<p class="note">Median error in $/week. Fold A is June and July 2026 adverts, Fold B is April and May 2026.</p>'
    body += ('<p class="note">Microburbs reports $86 on the frozen August and September adverts, which are different adverts, '
             "so it is not on this chart. Differences under about $1 are within noise.</p>")
    return section(None, "From $100 to now", lead, body)


def sec_lag(data, problems) -> str:
    x = data.get("index_bias")
    p = problems.get("index_bias", "field missing")
    before, after = num(dig(x, "overall", "bias_before")), num(dig(x, "overall", "bias_after"))
    intro = ("A naive estimate copies the last rent seen for a flat or its neighbours. That rent may be months old. "
             "When rents are rising, an old rent is a low rent, and the older it is the lower it runs.")
    parts = []
    if before is None or after is None:
        parts.append(pending_note("overall bias before and after the index. " + p))
    else:
        direction = f"ran {money(abs(before))} a week low" if before < 0 else f"was off by {money(before, True)} a week"
        parts.append(f"<p>Before the index, a typical estimate {esc(direction)}. After moving old rents forward to the "
                     f"target date with the index, the typical gap was {esc(money(after, True))} a week.</p>")
    ages = rows_of(dig(x, "by_age"))
    if ages:
        rows = [(str(r.get("age_band", "?")), [(num(r.get("bias_before")), "s1", "before index"),
                                               (num(r.get("bias_after")), "s2", "after index")]) for r in ages]
        parts.append("<h3>Typical gap by age of the rent used, $/week</h3>"
                     + legend([("s1", "Before index"), ("s2", "After index")]) + hbars(rows)
                     + '<p class="note">Below zero means the estimate was lower than the advert.</p>')
        parts.append(data_view(["Age of rent", "Adverts", "Before", "After"],
                               [(r.get("age_band", "?"), count(num(r.get("n"))), money(num(r.get("bias_before")), True),
                                 money(num(r.get("bias_after")), True)) for r in ages]))
    else:
        parts.append(pending_note("gap by age of rent. " + p))
    return section(1, "Why a naive estimate runs low", intro, "".join(parts))


def premium_rows(rows, key, label=str) -> list:
    return [(label(r[key]), num(r.get("premium")), num(r.get("low")), num(r.get("high")), num(r.get("n")))
            for r in rows if r.get(key) is not None]


def premium_block(title, head, rows, problem, rank=True) -> str:
    if not rows:
        return f"<h3>{esc(title)}</h3>" + pending_note(problem)
    shown = rows
    if rank:
        shown = sorted(rows, key=lambda r: -(r[4] or 0))[:MAX_BARS]
        shown.sort(key=lambda r: -(r[1] if r[1] is not None else -math.inf))
    note = f'<p class="note">Showing the {len(shown)} with the most adverts, of {len(rows)}. All are in the numbers below.</p>' \
        if len(shown) < len(rows) else ""
    return (f"<h3>{esc(title)}</h3>" + hbars([(r[0], [(r[1], "s1", "")]) for r in shown]) + note
            + data_view([head, "Premium", "Range", "Adverts"],
                        [(r[0], money(r[1]), span_text(r[2], r[3]), count(r[4])) for r in rows]))


def beds(v) -> str:
    n = num(v)
    if n is None:
        return str(v)
    return "Studio" if n == 0 else f"{n:.0f} bed"


def sec_parking(data, problems) -> str:
    x = data.get("parking")
    p = problems.get("parking", "field missing")
    overall = num(dig(x, "overall", "premium"))
    body = f"<p>Across all flats, a car space adds about <strong>{esc(money(overall))}</strong> a week.</p>" \
        if overall is not None else pending_note("overall parking premium. " + p)
    body += premium_block("By suburb, $/week", "Suburb", premium_rows(rows_of(dig(x, "by_suburb")), "suburb"),
                          "premium by suburb. " + p)
    bed_rows = sorted(rows_of(dig(x, "by_bedrooms")), key=lambda r: num(r.get("bedrooms")) or 0)
    body += premium_block("By bedroom count, $/week", "Bedrooms", premium_rows(bed_rows, "bedrooms", beds),
                          "premium by bedroom count. " + p, rank=False)
    lead = "Each flat gets two estimates, with and without a car space. The gap is what parking is worth in rent."
    note = text_of(dig(data.get("report_text"), "parking_note"))
    extra = f'<p class="note">{esc(note)}</p>' if note else ""
    return section(2, "What a car space is worth", lead, body + extra)


def sec_floor(data, problems) -> str:
    x = data.get("floor")
    p = problems.get("floor", "field missing")
    per = num(dig(x, "per_floor_premium"))
    body = f"<p>Each floor up is worth about <strong>{esc(money(per))}</strong> a week on average.</p>" \
        if per is not None else pending_note("average premium per floor. " + p)
    curve = [(num(r.get("floor")), num(r.get("premium")), num(r.get("low")), num(r.get("high")), num(r.get("n")))
             for r in rows_of(dig(x, "curve"))]
    curve = sorted((c for c in curve if c[0] is not None and c[1] is not None), key=lambda c: c[0])
    if len(curve) >= 2:
        ref = num(dig(x, "reference_floor"))
        base = f"floor {ref:.0f}" if ref is not None else "the lowest floor shown"
        steps = [None] + [(b[1] - a[1]) / (b[0] - a[0]) if b[0] != a[0] else None for a, b in zip(curve, curve[1:])]
        body += (f"<h3>Rent premium compared with {esc(base)}, $/week</h3>" + floor_chart(curve)
                 + data_view(["Floor", "Premium", "Per floor since last row", "Adverts"],
                             [(f"{c[0]:.0f}", money(c[1], True), money(s, True) if s is not None else "first row", count(c[4]))
                              for c, s in zip(curve, steps)]))
    else:
        body += pending_note("floor curve. " + p)
    lead = "Higher flats usually ask more. The curve shows whether each extra floor adds the same amount."
    return section(3, "What a higher floor is worth", lead, body)


def sec_error(data, problems) -> str:
    m = data.get("metrics")
    p = problems.get("metrics", "this method is not in metrics.json")

    def mae(k):
        return fold_metrics(method(m, k), "A")["mae"]

    rows = [("Suburb-only median (ours)", [(mae("suburb_median"), "s1", "ours")]),
            ("Microburbs valuation (their figure)", [(float(MICROBURBS_MAE), "s2", "published")]),
            ("Our baseline", [(mae("baseline"), "s1", "ours")]),
            ("Final model", [(mae("final"), "s1", "ours")])]
    body = ("<h3>Median error, $/week, lower is better</h3>"
            + legend([("s1", "Measured by us on Fold A"), ("s2", "Published by Microburbs")]) + hbars(rows))
    missing = [r[0] for r in rows if r[1][0][0] is None]
    if missing:
        body += pending_note(", ".join(missing) + ". " + p)
    body += ('<p class="note">Microburbs\' $86 is their own figure, not measured by us on the same adverts, so read it '
             "as a guide rather than a head-to-head result.</p>")
    body += table(["Published baseline", "Median error", "Within $100", "Typical bias"], PUBLISHED)
    return section(4, "Error by method", "Half of adverts are closer than the median error and half are further away.", body)


def sec_shap(data, problems) -> str:
    x = data.get("shap")
    unit = text_of(dig(x, "unit"))
    rows = [(str(r.get("label") or r.get("feature") or "?"), num(r.get("mean_abs_shap"))) for r in rows_of(dig(x, "top_features"))]
    rows = sorted(rows, key=lambda r: -(r[1] if r[1] is not None else -math.inf))[:MAX_BARS]
    if rows:
        fmt = money if unit == "$/week" else plain
        body = f"<h3>Average effect on a prediction, {esc(unit or 'units not stated')}</h3>" \
            + hbars([(a, [(b, "s1", "")]) for a, b in rows], fmt)
    else:
        body = pending_note("top features. " + problems.get("shap", "top_features missing"))
    finding = text_of(dig(data.get("report_text"), "surprising_finding"))
    body += "<h3>One surprising finding</h3>"
    body += f"<blockquote>{esc(finding)}</blockquote>" if finding \
        else pending_note("hand-written finding. Add surprising_finding to " + text_path().name)
    return section(5, "What drives the estimate", "SHAP measures how much each input moved a prediction.", body)


def sec_sources(data) -> str:
    items = ["<strong>Microburbs and the hackathon organisers</strong> supplied the flat list, past lease adverts, "
             "the mesh block homes and people file and the published baselines.",
             "<strong>G-NAF</strong>, the Geocoded National Address File, gives each flat its address identifier.",
             "<strong>ABS mesh blocks</strong>, from the Australian Bureau of Statistics, are the small areas used for local data."]
    if dig(data.get("metrics"), "sources", "nsw_bond_used") is True:
        items.append(f"<strong>NSW rental bond data</strong>. {esc(NSW_ATTRIBUTION)}")
    return section(None, "Data and sources", None, "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"
                   + "<p>Supplied data and row-level results are not published here. This report holds only totals and averages.</p>")


def sec_method(data, problems) -> str:
    summary = text_of(dig(data.get("metrics"), "method_summary"))
    final = f"<p>{esc(summary)}</p>" if summary \
        else pending_note("final model description. " + problems.get("metrics", "method_summary missing"))
    steps = ["Find the most specific past rent for each flat: its own, then its building, then its block.",
             "Move that rent forward to the target date with a rent index, so an old rent is not read as a current one.",
             "Train on rolling past dates. Each training row sees only what was known two months earlier, never its own advert.",
             "Predict twice for each flat, once with a car space and once without."]
    body = final + "<ol>" + "".join(f"<li>{esc(s)}</li>" for s in steps) + "</ol>"
    body += f"<h3>How the dates are split</h3>{split_diagram()}"
    body += (f'<p class="note">The left bar in each row is what the model may learn from. The right bar is what it is tested on '
             f"and never sees. Final estimates use data up to {esc(d(C.AS_OF))} and aim at {esc(d(C.SCORE_MID))}, "
             f"the middle of the scoring window.</p>")
    return section(None, "Method", None, body)


def sec_results(data, problems) -> str:
    methods = rows_of(dig(data.get("metrics"), "methods"))
    if not methods:
        return section(None, "Held-out results", None, pending_note("results table. " + problems.get("metrics", "methods missing")))
    names = {"A": f"Fold A: adverts {d(C.FOLDS['A']['start'])} to {d(C.FOLDS['A']['end'])}. Not used for any choice.",
             "B": f"Fold B: adverts {d(C.FOLDS['B']['start'])} to {d(C.FOLDS['B']['end'])}. Used for tuning, so it may flatter the model."}
    body = ""
    for k, cap in names.items():
        rows = []
        for x in methods:
            f = fold_metrics(x, k)
            cells = (money(f["mae"]), pct(f["within100"]), plain(f["r2"]), money(f["bias"], True))
            rows.append((x.get("label") or x.get("key") or "?", count(f["n"]), *["not logged" if c == PENDING else c for c in cells]))
        body += table(["Method", "Adverts", "Median error", "Within $100", "R2", "Bias"], rows, cap)
    body += '<p class="note">Bias is the median of prediction minus advert, so below zero means low.</p>'
    return section(None, "Held-out results", None, body)


def sec_list(title, data, key) -> str:
    raw = dig(data.get("report_text"), key)
    items = [text_of(i) for i in raw if text_of(i)] if isinstance(raw, list) else []
    body = "<ul>" + "".join(f"<li>{esc(i)}</li>" for i in items) + "</ul>" if items \
        else pending_note(f"hand-written list. Add {key} to {text_path().name}")
    return section(None, title, None, body)


def sec_limits(data=None) -> str:
    raw = dig((data or {}).get("report_text"), "limitations")
    extra = [text_of(i) for i in raw if text_of(i)] if isinstance(raw, list) else []
    items = ["These are asking rents from adverts, not the rent finally agreed.",
             "Flats that are advertised often are better covered than flats that rarely change hands.",
             "The parking and floor premiums are patterns in adverts. Other differences between flats may be mixed in.",
             "Estimates describe a typical flat of that kind. Renovations, views and condition are mostly invisible to the data.",
             "The test periods cover four months of 2026, so a sudden market change could behave differently.",
             "Our folds may flatter us. Flats with no earlier advert keep input fields that may come from the test advert itself, and fold adverts number about 1,300 a month against 1,699 scored over two months, so the scored adverts may be a different mix.",
             "Fold A was not perfectly untouched. No setting was tuned on it, but the check that found the input leak looked at its adverts, and it was read again for the final LightGBM and for the blend.",
             "We have no floor area or building age, so bigger or newer flats that have parking can inflate the premium. Treat it as an upper estimate. The two premium methods tied on median error on Fold B.",
             "Microburbs' $86 was measured on the frozen August and September adverts, not on our folds, so the comparison is indicative only."]
    return section(None, "Limitations", None, "<ul>" + "".join(f"<li>{esc(i)}</li>" for i in items + extra) + "</ul>")


CSS = """
:root{color-scheme:light;--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3de;--panel:#f3f2ee;--s1:#2a78d6;--s2:#eb6834;--warn-bg:#fff4d6;--warn-ink:#5c4300;--alarm:#a11d1d}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3a37;--panel:#262624;--s1:#3987e5;--s2:#d95926;--warn-bg:#3b2f0c;--warn-ink:#f6dd9a}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:760px;margin:0 auto;padding:16px}
h1{font-size:1.9rem;line-height:1.2;margin:16px 0 8px}h2{font-size:1.35rem;line-height:1.25;margin:0 0 8px}h3{font-size:1rem;margin:20px 0 8px}
section{border-top:1px solid var(--line);padding:24px 0}
.lead{color:var(--ink2);margin-top:0}.note{color:var(--ink2);font-size:.875rem}
.banner{background:var(--alarm);color:#fff;padding:16px;text-align:center;font-weight:700;font-size:1.25rem;line-height:1.3}
.banner small{display:block;font-size:.9rem;font-weight:400;margin-top:4px}
.pending{background:var(--warn-bg);color:var(--warn-ink);padding:8px 12px;border-radius:6px;font-size:.9rem;overflow-wrap:anywhere}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin:16px 0}
.tile{background:var(--panel);border-radius:8px;padding:12px}.tile b{display:block;font-size:1.6rem;line-height:1.2}.tile span{font-size:.85rem;color:var(--ink2)}
.tile.is-pending{background:var(--warn-bg)}.tile.is-pending b,.tile.is-pending span{color:var(--warn-ink)}
.legend{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:.875rem;color:var(--ink2);margin-bottom:8px}
.key{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.key.s1,.bar.s1{background:var(--s1)}.key.s2,.bar.s2{background:var(--s2)}
.barrow{display:grid;grid-template-columns:minmax(0,34%) minmax(0,1fr);gap:8px;align-items:center;padding:4px 0}
.lab{font-size:.875rem;overflow-wrap:anywhere}
.line{display:grid;grid-template-columns:minmax(0,1fr) 68px;gap:6px;align-items:center;margin:2px 0}
.track{position:relative;height:14px}.track:before{content:"";position:absolute;left:var(--zero);top:-2px;bottom:-2px;width:1px;background:var(--ink2)}
.bar{position:absolute;top:0;bottom:0}.bar.pos{border-radius:0 4px 4px 0}.bar.neg{border-radius:4px 0 0 4px}
.val{font-size:.85rem;text-align:right;font-variant-numeric:tabular-nums}.val.is-pending{color:var(--warn-ink);background:var(--warn-bg);border-radius:4px;padding:0 4px}
.plot{display:block;width:100%;height:auto}
.plot text{font-family:inherit}.tick{font-size:12px;fill:var(--ink2)}.rowlab{font-size:13px;font-weight:600;fill:var(--ink)}
.grid{stroke:var(--line);stroke-width:1}.zero{stroke:var(--ink2);stroke-width:1}
.curve{fill:none;stroke:var(--s1);stroke-width:2}.dot{fill:var(--s1);stroke:var(--bg);stroke-width:2}.band{fill:var(--s1);opacity:.18}
.past{fill:var(--s1)}.held{fill:var(--s2)}.inbar{font-size:12px;fill:#fff}.inbar.dark{fill:#0b0b0b}
details{margin-top:8px;font-size:.9rem}summary{cursor:pointer;color:var(--ink2)}
.tablewrap{overflow-x:auto;margin:12px 0}table{border-collapse:collapse;width:100%;font-size:.875rem}
caption{text-align:left;font-weight:600;padding-bottom:6px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
blockquote{margin:0;padding:8px 12px;border-left:3px solid var(--s1);background:var(--panel)}
footer{color:var(--ink2);font-size:.85rem;padding:16px 0}

@keyframes rise{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
@keyframes grow{from{transform:scaleX(0)}to{transform:scaleX(1)}}
@keyframes reveal{from{opacity:.15;transform:translateY(26px)}to{opacity:1;transform:none}}
.tile{animation:rise .6s ease-out both}.tile:nth-child(2){animation-delay:.15s}.tile:nth-child(3){animation-delay:.3s}
.bar{transform-origin:left;animation:grow 1.1s cubic-bezier(.2,.7,.2,1) both}.bar.neg{transform-origin:right}
.barrow:nth-child(2) .bar{animation-delay:.12s}.barrow:nth-child(3) .bar{animation-delay:.24s}.barrow:nth-child(4) .bar{animation-delay:.36s}.barrow:nth-child(5) .bar{animation-delay:.48s}.barrow:nth-child(6) .bar{animation-delay:.6s}.barrow:nth-child(7) .bar{animation-delay:.72s}
@supports (animation-timeline:view()){section{animation:reveal linear both;animation-timeline:view();animation-range:entry 0% entry 35%}.bar{animation-timeline:view();animation-duration:auto;animation-delay:0s!important;animation-range:entry 10% entry 55%}}
@media (prefers-reduced-motion:reduce){.tile,.bar,section{animation:none!important}}
"""

BANNERS = {
    "mock": ("SYNTHETIC NUMBERS. NOT REAL RESULTS.", "This page was built from mock data for testing. Do not quote anything on it."),
    "fixture": ("FAKE LAYOUT FIXTURE. EVERY NUMBER IS MADE UP.", "This page only tests the layout. Do not quote anything on it."),
}


def render(data: dict, problems: dict, mode: str) -> str:
    banner = ""
    if mode in BANNERS:
        a, b = BANNERS[mode]
        banner = f'<div class="banner" role="alert">{esc(a)}<small>{esc(b)}</small></div>'
    flag = "" if mode == "real" else f"[{mode.upper()}] "
    intro = (f'<h1>Fair weekly rent for every City of Sydney flat</h1><p class="lead">We estimated the weekly asking rent for all '
             f"{C.N_INPUT:,} City of Sydney flats, with and without parking, as at {esc(d(C.AS_OF))}. "
             f"This page shows what we learned first, then how we did it.</p>")
    parts = [intro, tiles(data), sec_journey(data, problems), sec_lag(data, problems), sec_parking(data, problems), sec_floor(data, problems),
             sec_error(data, problems), sec_shap(data, problems), sec_sources(data), sec_method(data, problems),
             sec_results(data, problems), sec_list("What worked", data, "what_worked"),
             sec_list("What didn't", data, "what_didnt"), sec_limits(data)]
    foot = f"<footer>Built {esc(datetime.now(timezone.utc).strftime('%-d %b %Y %H:%M UTC'))}.</footer>"
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{esc(flag)}Sydney Flat Rents</title><style>{CSS}</style></head>"
            f"<body>{banner}<main>{''.join(parts)}{foot}</main>{banner}</body></html>")


def make_fixture() -> dict:
    """Clearly fake values in every shape, for layout only. Repeating digits make them easy to spot."""
    def fold(a):
        return dict(n=1111, mae=a, within100=0.1111, r2=0.1111, bias=-11.0)

    base = dict(provenance={"mode": "fixture"}, fixture=True)
    sub = [dict(suburb=f"Fake Suburb {c}", premium=11.0 * (i + 1), low=11.0 * i, high=11.0 * (i + 2), n=111 * (20 - i))
           for i, c in enumerate("ABCDEFGHIJKLMNOPQR")]
    ages = [("0 to 3 months", -22.0, 11.0), ("3 to 12 months", -44.0, -11.0), ("Over 12 months", -88.0, -22.0)]
    return dict(
        metrics=dict(base, method_summary="Fake sentence where the final model is named.", sources=dict(nsw_bond_used=True),
                     methods=[dict(key=k, label=f"Fake {k}", folds=dict(A=fold(v), B=fold(v + 11)))
                              for k, v in [("suburb_median", 111.0), ("baseline", 99.0), ("final", 55.5)]]),
        index_bias=dict(base, fold="A", overall=dict(bias_before=-77.0, bias_after=11.0),
                        by_age=[dict(age_band=b, n=1111, bias_before=v, bias_after=w) for b, v, w in ages]),
        parking=dict(base, overall=dict(premium=77.0, low=66.0, high=88.0, n=1111), by_suburb=sub,
                     by_bedrooms=[dict(bedrooms=b, premium=22.0 * (b + 1), low=11.0 * (b + 1), high=33.0 * (b + 1), n=111)
                                  for b in range(4)]),
        floor=dict(base, per_floor_premium=5.55, reference_floor=1,
                   curve=[dict(floor=f, premium=5.55 * (f - 1), low=4.44 * (f - 1), high=6.66 * (f - 1), n=111)
                          for f in range(1, 32, 3)]),
        shap=dict(base, unit="$/week",
                  top_features=[dict(feature=f"fake_{i}", label=f"Fake feature {i}", mean_abs_shap=111.0 / i) for i in range(1, 9)]),
        report_text=dict(base, surprising_finding="Fake finding. The team writes the real one by hand.",
                         what_worked=["Fake item that worked."], what_didnt=["Fake item that did not."]),
    )


def build(fixture=False) -> Path:
    if fixture:
        if not C.MOCK:
            raise Refused("--fixture needs FLATS_MOCK=1. A real run never uses the fixture")
        data, problems, mode, out = make_fixture(), {}, "fixture", C.OUT_DIR / "report_fixture.html"
    else:
        data, problems = load_inputs()
        if C.MOCK:
            mode, out = "mock", C.OUT_DIR / "report.html"
        else:
            require_real(data)
            mode, out = "real", C.ROOT / "report.html"
    page = render(data, problems, mode)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(page, encoding="utf-8")
    tmp.replace(out)
    for name, reason in problems.items():
        print(f"pending: {name}: {reason}")
    return out


# ---------------------------------------------------------------- pitch
def whole(x) -> str:
    return f"{math.floor(abs(x) + 0.5):,}"


def pitch_values(data: dict) -> dict:
    """Fill values, leaving out anything missing or that would make the script's wording false."""
    f = fold_metrics(method(data.get("metrics"), "final"), "A")
    before = num(dig(data.get("index_bias"), "overall", "bias_before"))
    after = num(dig(data.get("index_bias"), "overall", "bias_after"))
    park, floor = num(dig(data.get("parking"), "overall", "premium")), num(dig(data.get("floor"), "per_floor_premium"))
    v = {}
    if before is not None and after is not None and before < 0 and abs(after) < abs(before):  # "ran low", "cut that"
        v.update(lag_low_before=whole(before), lag_gap_after=whole(after))
    if f["mae"] is not None:
        v["final_median_error"] = whole(f["mae"])
    if f["within100"] is not None:
        v["final_within_100_pct"] = f"{f['within100'] * 100:.0f}"
    if park is not None and park > 0:
        v["parking_premium"] = whole(park)
    if floor is not None and floor > 0:  # the script says "adds"; cents matter at this size
        v["floor_premium"] = f"{floor:.2f}".removesuffix(".00")
    return v


def fill_pitch() -> Path:
    data, _ = load_inputs()
    require_real(data)
    template = (C.ROOT / "PITCH_SCRIPT.md").read_text(encoding="utf-8")
    values = pitch_values(data)
    missing = sorted(set(re.findall(r"\{\{(\w+)\}\}", template)) - set(values))
    if missing:
        raise Refused("no usable value for: " + ", ".join(missing))
    out = C.ROOT / "PITCH_SCRIPT.filled.md"
    out.write_text(re.sub(r"\{\{(\w+)\}\}", lambda m: values[m.group(1)], template), encoding="utf-8")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", action="store_true", help="fake numbers for layout testing, needs FLATS_MOCK=1")
    ap.add_argument("--pitch", action="store_true", help="fill PITCH_SCRIPT.md from real metrics")
    a = ap.parse_args(argv)
    try:
        print("wrote", fill_pitch() if a.pitch else build(a.fixture))
        return 0
    except Refused as e:
        print(f"REFUSED: {e}. Nothing was written.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
