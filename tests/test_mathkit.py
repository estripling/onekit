from typing import Tuple

import pytest

import onekit.mathkit as mk


@pytest.mark.parametrize(
    "n, expected",
    [
        (-2, None),
        (-1, None),
        (0, None),
        (1, (1,)),
        (2, (2, 1)),
        (4, (4, 2, 1)),
        (7, (7, 22, 11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1)),
        (11, (11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1)),
        (12, (12, 6, 3, 10, 5, 16, 8, 4, 2, 1)),
    ],
)
def test_collatz(n: int, expected: Tuple[int]):
    if n > 0:
        actual = tuple(mk.collatz(n))
        assert actual == expected
    else:
        with pytest.raises(ValueError):
            tuple(mk.collatz(n))
