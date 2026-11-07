"""Leak guards for API data. No network: _fetch is replaced, and the cache points at a temporary folder."""
import numpy as np
import pandas as pd
import pytest

from src import api
from src import api_features as AF


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "CACHE_DIR", tmp_path)
    return tmp_path


def _reply(monkeypatch, body):
    monkeypatch.setattr(api, "_fetch", lambda path, params: body)


def _adverts(*dates):
    return {"data": {"meta": {}, "results": [{"gnaf_pid": "G", "listing_date": d, "weekly_rent": 500} for d in dates]}}


@pytest.mark.parametrize("path", ["suburbs/Sydney/listings/for-rent", "properties/G/rent-history-latest", "properties/G/rent-history",
                                  "properties/G/surroundings/listing-history-signals", "properties/G/street/profile",
                                  "suburbs/Sydney/market/vacancy-map", "suburbs/Sydney/market/rent-map", "properties/G/context/mesh-blocks",
                                  "properties/G/valuation/avm", "suburbs/Sydney/forecast/sale"])
def test_banned_endpoints_are_refused(cache, path):
    with pytest.raises(PermissionError):
        api.get(path)


@pytest.mark.parametrize("params", [{}, {"to_date": "2026-08-01"}])
def test_rent_history_needs_a_bound(cache, monkeypatch, params):
    _reply(monkeypatch, _adverts("2026-07-01"))
    with pytest.raises(api.LeakError):
        api.get("suburbs/Sydney/listings/rent-history", params)


@pytest.mark.parametrize("dates", [("2026-07-31", "2026-08-01"), ("2026-07-01", None)])
def test_later_or_undated_adverts_discard_the_response(cache, monkeypatch, dates):
    _reply(monkeypatch, _adverts(*dates))
    with pytest.raises(api.LeakError):
        api.get("suburbs/Sydney/listings/rent-history", {"to_date": "2026-07-31"})
    assert not list(cache.rglob("*.json"))


def test_series_are_cut_before_caching(cache, monkeypatch):
    _reply(monkeypatch, {"data": {"as_of": "2026-10-01", "unit": [{"date": "2026-07-26", "value": 1}, {"date": "2026-08-02", "value": 2}]}})
    path = "suburbs/Sydney/market/median-rent-series"
    api.get(path)
    assert "2026-08-02" not in next(cache.rglob("*.json")).read_text() and "2026-10-01" not in next(cache.rglob("*.json")).read_text()


def test_sale_history_is_cut_before_caching(cache, monkeypatch):
    _reply(monkeypatch, {"data": [{"date": "2026-09-01", "price": 2, "type": "Sold"}, {"date": "2020-01-01", "price": 1, "type": "Sold"}]})
    assert [x["date"] for x in api.get("properties/G/sale-history")["data"]] == ["2020-01-01"]


def test_edge_403_is_not_cached_as_outside_coverage(cache, monkeypatch):
    def refuse(path, params):
        raise RuntimeError("403 from the edge")
    monkeypatch.setattr(api, "_fetch", refuse)
    with pytest.raises(RuntimeError):
        api.get("ping")
    assert not list(cache.rglob("*.json"))


def test_sales_never_reach_past_a_row_origin(monkeypatch):
    sales = pd.DataFrame({"gnaf_id": ["A", "A", "B"], "date": pd.to_datetime(["2020-01-01", "2026-03-01", "2026-05-01"]),
                          "price": [500_000.0, 900_000.0, 700_000.0]})
    flats = pd.DataFrame({"gnaf_id": ["A", "B"], "building_key": ["K", "K"]})
    monkeypatch.setattr(AF, "sales_table", lambda: sales)
    monkeypatch.setattr(AF, "load_input", lambda: flats)
    frame = pd.DataFrame({"gnaf_id": ["A", "A", "B"], "building_key": ["K"] * 3,
                          "origin": pd.to_datetime(["2026-01-31", "2026-03-31", "2026-03-31"])})
    out = AF.add_sales(frame)
    assert np.isclose(out.sale_log_price[0], np.log(500_000)) and out.sale_n[0] == 1
    assert np.isclose(out.sale_log_price[1], np.log(900_000))
    assert np.isnan(out.sale_log_price[2])  # B's own May sale is later than the origin
    assert out.bldg_sale_n[2] == 1  # only A's March sale: the 2020 sale is over five years back, B's is in the future


def test_market_reads_only_up_to_the_origin(monkeypatch):
    t = pd.DataFrame({"suburb": "S", "date": pd.to_datetime(["2025-01-31", "2025-09-30", "2025-12-31", "2026-03-31", "2026-04-30"]),
                      "value": [100.0, 110.0, 120.0, 130.0, 9999.0]})
    monkeypatch.setattr(AF, "series", lambda tail, field="unit": t)
    out = AF.add_market(pd.DataFrame({"suburb": ["S"], "origin": pd.to_datetime(["2026-03-31"])}))
    assert np.isclose(out.mkt_rent_chg_3m[0], np.log(130 / 120))


def test_point_in_polygon_handles_holes():
    ring = lambda a, b: [[a, a], [b, a], [b, b], [a, b], [a, a]]
    feats = [{"properties": {"h": 10}, "geometry": {"type": "Polygon", "coordinates": [ring(0, 10), ring(4, 6)]}}]
    got = AF._at_points(np.array([1.0, 5.0, 11.0]), np.array([1.0, 5.0, 1.0]), feats, "h")
    assert got[0] == 10 and np.isnan(got[1]) and np.isnan(got[2])


def test_segment_distance():
    d = AF._dist_to_segments(np.array([[0.0, 3.0], [10.0, 0.0]]), np.array([[-1.0, 0.0]]), np.array([[1.0, 0.0]]))
    assert np.allclose(d, [3, 9])
