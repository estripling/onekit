from typing import (
    Tuple,
    Union,
)

import pytest
import toolz

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


@pytest.mark.parametrize(
    "x, expected",
    [
        (0, 0.0),
        (0.1, 0.0),
        (1, 1.0),
        (10, 2.0),
        (100, 3.0),
        (1_000, 4.0),
        (10_000, 5.0),
        (100_000, 6.0),
        (1_000_000, 7.0),
        (0.2, 0.30102999566398125),
        (2, 1.3010299956639813),
        (20, 2.3010299956639813),
        (200, 3.3010299956639813),
        (2_000, 4.3010299956639813),
        (-0.5, 0.6989700043360187),
        (-5, 1.6989700043360187),
        (-50, 2.6989700043360187),
        (-500, 3.6989700043360187),
        (-5_000, 4.6989700043360187),
    ],
)
def test_digitscale(x: Union[int, float], expected: float):
    for v in (-x, x):
        actual = mk.digitscale(v)
        assert actual == expected


def test_fibonacci():
    actual = tuple(toolz.take(16, mk.fibonacci()))
    expected = (0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610)
    assert actual == expected


@pytest.mark.parametrize("x", [-1, 0, 1, 2, 3, 3.14, 4, 5, 6, 7, 8, 9, 10, 11.0])
@pytest.mark.parametrize("n", [2, 5])
def test_isdivisible(x: Union[int, float], n: int):
    actual = mk.isdivisible(x, n)
    expected = x % n == 0
    assert actual == expected


@pytest.mark.parametrize("x", [-1, 0, 1, 2, 3, 3.14, 4, 5, 6, 7, 8, 9, 10, 11.0])
def test_iseven(x: Union[int, float]):
    actual = mk.iseven(x)
    expected = x % 2 == 0
    assert actual == expected


@pytest.mark.parametrize("x", [-1, 0, 1, 2, 3, 3.14, 4, 5, 6, 7, 8, 9, 10, 11.0])
def test_isodd(x: Union[int, float]):
    actual = mk.isodd(x)
    is_even_number = x % 2 == 0
    expected = not is_even_number
    assert actual == expected


@pytest.mark.parametrize(
    "x, expected",
    [
        (-2.0, -1),
        (-1, -1),
        (-0, 0),
        (1, 1),
        (2.0, 1),
    ],
)
def test_sign(x: Union[int, float], expected: int):
    actual = mk.sign(x)
    assert isinstance(actual, int)
    assert actual == expected
