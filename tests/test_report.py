import json
import re
from pathlib import Path

import pytest

from src import config as C
from src import report as R

REAL = {"provenance": {"mode": "real"}}
GOOD = {
    "metrics.json": {**REAL, "methods": [{"key": "final", "folds": {"A": {"mae": 70.5, "within100": 0.66, "r2": 0.8, "bias": -3}}}]},
    "index_bias.json": {**REAL, "overall": {"bias_before": -40, "bias_after": 4}},
    "parking.json": {**REAL, "overall": {"premium": 60}},
    "floor.json": {**REAL, "per_floor_premium": 3.5},
}
TEMPLATE = Path(R.__file__).parents[1] / "notes" / "PITCH_SCRIPT.md"


@pytest.fixture
def env(tmp_path, monkeypatch):
    (tmp_path / "out").mkdir()
    monkeypatch.setattr(C, "ROOT", tmp_path)
    monkeypatch.setattr(C, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(C, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(C, "MOCK_TRUTH_CSV", tmp_path / "data" / "_frozen_truth.csv")
    monkeypatch.setattr(C, "MOCK", False)
    (tmp_path / "notes").mkdir(exist_ok=True); (tmp_path / "notes" / "PITCH_SCRIPT.md").write_text(TEMPLATE.read_text())
    return tmp_path


def put(env, files):
    for name, value in files.items():
        (env / "out" / name).write_text(json.dumps(value))


def test_empty_real_run_is_all_pending_with_no_invented_numbers(env):
    page = R.build().read_text()
    assert (env / "report.html").exists()
    assert page.count('class="pending"') >= 10
    assert set(re.findall(r"\$\d+", page)) <= {"$86", "$100", "$93", "$61"}  # published figures and the threshold
    assert "SYNTHETIC" not in page and "FAKE" not in page


@pytest.mark.parametrize("bad", [{"provenance": {"mode": "mock"}}, {"provenance": {"mode": "fixture"}}, {"x": 1},
                                 {**REAL, "fixture": True}])
def test_real_run_refuses_unstamped_mock_and_fixture_inputs(env, bad):
    put(env, {"parking.json": bad})
    with pytest.raises(R.Refused):
        R.build()
    assert not (env / "report.html").exists()


def test_real_run_refuses_mock_data_folder(env):
    (env / "data").mkdir()
    (env / "data" / "_frozen_truth.csv").write_text("x")
    with pytest.raises(R.Refused):
        R.build()


def test_mock_run_stays_in_out_dir_with_banner(env, monkeypatch):
    monkeypatch.setattr(C, "MOCK", True)
    path = R.build()
    assert path == env / "out" / "report.html" and not (env / "report.html").exists()
    assert "SYNTHETIC NUMBERS" in path.read_text()


def test_fixture_needs_mock_and_writes_no_inputs(env, monkeypatch):
    with pytest.raises(R.Refused):
        R.build(fixture=True)
    monkeypatch.setattr(C, "MOCK", True)
    assert "FAKE LAYOUT FIXTURE" in R.build(fixture=True).read_text()
    assert [p.name for p in (env / "out").iterdir()] == ["report_fixture.html"]


def test_section_order_self_contained_and_no_em_dash(env, monkeypatch):
    monkeypatch.setattr(C, "MOCK", True)
    page = R.build(fixture=True).read_text()
    heads = re.findall(r"<h2>(.*?)</h2>", page)
    assert heads[0] == "From $100 to now"
    assert [h[:2] for h in heads[1:6]] == ["1.", "2.", "3.", "4.", "5."]
    assert heads[6:] == ["Data and sources", "Method", "Held-out results", "What worked", "What didn&#x27;t", "Limitations"]
    assert not re.search(r"<script|<link|<img|<iframe|src=|href=|url\(|https?://", page)
    assert "—" not in page


def test_nsw_credit_only_when_flagged(env):
    put(env, {"metrics.json": {**REAL, "sources": {"nsw_bond_used": "yes"}}})
    assert R.NSW_ATTRIBUTION not in R.html.unescape(R.build().read_text())
    put(env, {"metrics.json": {**REAL, "sources": {"nsw_bond_used": True}}})
    assert R.NSW_ATTRIBUTION in R.html.unescape(R.build().read_text())


def test_bad_values_become_pending_and_text_is_escaped(env):
    (env / "out" / "floor.json").write_text("{not json")
    put(env, {"metrics.json": {**REAL, "method_summary": "<b>x</b>", "methods": [
        {"key": "final", "label": "L", "folds": {"A": {"mae": float("nan"), "r2": "0.9", "within100": 66}}}]}})
    page = R.build().read_text()
    assert "floor.json could not be read" in page
    assert "&lt;b&gt;x" in page and "66.00%" in page and "NaN" not in page and "0.900" not in page


def test_pitch_fills_from_real_only(env):
    put(env, GOOD)
    text = R.fill_pitch().read_text()
    assert "{{" not in text and "$71 a week" in text and "66%" in text and "$3.50" in text


def test_pitch_refuses_mock(env, monkeypatch):
    put(env, GOOD)
    monkeypatch.setattr(C, "MOCK", True)
    assert R.main(["--pitch"]) == 2
    assert not (env / "notes" / "PITCH_SCRIPT.filled.md").exists()


@pytest.mark.parametrize("change", [{"metrics.json": REAL},
                                    {"index_bias.json": {**REAL, "overall": {"bias_before": 5, "bias_after": 1}}},
                                    {"floor.json": {**REAL, "per_floor_premium": -1}}])
def test_pitch_refuses_gaps_and_false_wording(env, change):
    put(env, {**GOOD, **change})
    with pytest.raises(R.Refused):
        R.fill_pitch()


def test_pitch_length():
    words = TEMPLATE.read_text().split("## Script")[1].split()
    assert 100 <= len(words) <= 120


def test_journey_shows_steps_and_a_step_that_got_worse():
    steps = [dict(step="A", change="first", foldA=100.0, within100=.5, foldB=95.0), dict(step="B", change="second", foldA=50.0, within100=.7, foldB=60.0),
             dict(step="C", change="honest", foldA=50.25, within100=.7, foldB=59.0)]
    page = R.sec_journey({"report_text": {"journey": steps}}, {})
    assert "1. A" in page and "3. C" in page and "-$50.00" in page and "+$0.25" in page


def test_journey_without_steps_is_pending():
    assert "pending" in R.sec_journey({}, {}).lower()
