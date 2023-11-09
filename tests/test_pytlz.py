import datetime as dt

import pytest

import onekit as ok


@pytest.mark.parametrize(
    "d, expected",
    [
        (dt.date(2022, 1, 1), "2022-01-01"),
        (dt.date(2022, 1, 31), "2022-01-31"),
    ],
)
def test_to_str(d, expected):
    actual = ok.date_to_str(d)
    assert actual == expected


@pytest.mark.parametrize(
    "n, expected",
    [
        (1, "1"),
        (10, "10"),
        (100, "100"),
        (1000, "1_000"),
        (1000000, "1_000_000"),
        (100000.0, "100_000.0"),
    ],
)
def test_num_to_str(n, expected):
    actual = ok.num_to_str(n)
    assert actual == expected
