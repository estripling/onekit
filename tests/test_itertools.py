import calendar
import itertools
import math

import pytest
from toolz import curried

from onekit import pytlz


class TestItertoolsAPI:
    def test_count__basic(self):
        with pytest.raises(TypeError):
            itertools.count(start="a_string")

        with pytest.raises(TypeError):
            itertools.count(start=1, step="a_string")

        count_five = curried.compose(tuple, curried.take(5), itertools.count)

        assert count_five() == (0, 1, 2, 3, 4)
        assert count_five(1) == (1, 2, 3, 4, 5)
        assert count_five(10, 3) == (10, 13, 16, 19, 22)

        assert count_five(start=1) == (1, 2, 3, 4, 5)
        assert count_five(start=1, step=0) == (1, 1, 1, 1, 1)
        assert count_five(start=1, step=2) == (1, 3, 5, 7, 9)
        assert count_five(start=1, step=-1) == (1, 0, -1, -2, -3)
        assert count_five(start=1, step=-2) == (1, -1, -3, -5, -7)

        assert count_five(start=-2) == (-2, -1, 0, 1, 2)
        assert count_five(start=-2, step=0) == (-2, -2, -2, -2, -2)
        assert count_five(start=-2, step=2) == (-2, 0, 2, 4, 6)
        assert count_five(start=-2, step=-1) == (-2, -3, -4, -5, -6)
        assert count_five(start=-2, step=-2) == (-2, -4, -6, -8, -10)

        assert count_five(start=1, step=0.5) == (1.0, 1.5, 2.0, 2.5, 3.0)

    def test_count__advanced(self):
        takewhile = curried.curry(itertools.takewhile)

        actual = curried.pipe(itertools.count(-2), takewhile(lambda x: x <= 4), tuple)
        assert actual == (-2, -1, 0, 1, 2, 3, 4)

        actual = curried.pipe(
            itertools.count(0, 0.1),
            curried.map(pytlz.signif),
            takewhile(lambda x: x <= 1),
            tuple,
        )
        expected = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        assert actual == expected

        power_base = curried.curry(math.pow)
        power_exponents = curried.pipe(
            itertools.count(-5),
            takewhile(lambda x: x <= 4),
            tuple,
        )

        actual = tuple(map(power_base(2), power_exponents))
        expected = (0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
        assert actual == expected

        actual = tuple(map(power_base(10), power_exponents))
        expected = (1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
        assert actual == expected

        actual = curried.pipe(
            itertools.count(0),
            curried.filter(pytlz.iseven),
            curried.take(5),
            tuple,
        )
        expected = (0, 2, 4, 6, 8)
        assert actual == expected

        actual = curried.pipe(
            itertools.count(0),
            curried.filter(pytlz.isodd),
            curried.take(5),
            tuple,
        )
        expected = (1, 3, 5, 7, 9)
        assert actual == expected

        actual = curried.pipe(itertools.count(0, step=-4), curried.take(5), tuple)
        expected = (0, -4, -8, -12, -16)
        assert actual == expected

        actual = curried.pipe(
            itertools.count(1800),
            curried.filter(calendar.isleap),
            curried.filter(lambda year: str(year).endswith("0")),
            curried.take_nth(2),
            curried.take(10),
            tuple,
        )
        expected = (1820, 1860, 1920, 1960, 2000, 2040, 2080, 2140, 2180, 2240)
        assert actual == expected
