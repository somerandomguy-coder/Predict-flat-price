import numpy as np
import pandas as pd
import pytest

from src.load import parse_date
from src.validate import check_submission, round_half, score


def rows(**k):
    base = dict(rent=[500.0, 600.0, 700.0], rent_with_car_space=[550.0, 650.0, 800.0],
                rent_without_car_space=[500.0, 500.0, 700.0], advert_car_space=[1, np.nan, np.nan],
                input_car_space=[0, 1, np.nan])
    return pd.DataFrame({**base, **k})


def test_advert_parking_wins_then_input_then_excluded():
    s = score(rows())
    assert s["n"] == 2 and s["n_excluded"] == 1
    assert s["mae"] == 50 and s["bias"] == 50  # 550 vs 500, and 650 vs 600 via input parking


def test_within_100_is_inclusive():
    d = rows(rent=[450.0, 600.0, 700.0], rent_with_car_space=[550.0, 600.0, 700.0])
    assert score(d)["within100"] == 1.0


def test_blank_scored_answer_is_worse_than_any_miss():
    d = rows(rent_with_car_space=[np.nan, np.nan, 800.0])
    s = score(d)
    assert s["n_blank"] == 2 and s["mae"] == np.inf and s["within100"] == 0


def test_round_half():
    assert list(round_half([100.2, 100.3, 100.8])) == [100.0, 100.5, 101.0]


def test_sentinel_date_is_unknown():
    assert parse_date(pd.Series(["0001-01-01"])).isna().all()


@pytest.mark.parametrize("bad", [
    dict(rent_with_car_space=[1.0, 2.0], rent_without_car_space=[2.0, 2.0]),
    dict(rent_with_car_space=[2.25, 2.0], rent_without_car_space=[2.0, 2.0]),
    dict(rent_with_car_space=[np.nan, 2.0], rent_without_car_space=[2.0, 2.0]),
])
def test_submission_rejects(bad):
    with pytest.raises(AssertionError):
        check_submission(pd.DataFrame({"gnaf_id": ["a", "b"], **bad}), pd.Series(["a", "b"]))
