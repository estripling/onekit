import pytest

import onekit


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
    actual = onekit.num_to_str(n)
    assert actual == expected
