import datetime as dt
import functools
import math
import os
import random
from typing import Tuple

import pytest
import toolz

from onekit import pytlz


@pytest.mark.parametrize(
    "x, func, expected",
    [
        (3, all, False),
        (5, all, False),
        (60, all, True),
        (9, all, False),
        (3, any, True),
        (5, any, True),
        (60, any, True),
        (9, any, True),
        (12, any, True),
        (13, any, False),
    ],
)
def test_are_predicates_true(x, func, expected):
    predicates = [
        lambda x: x % 3 == 0,
        lambda x: x % 5 == 0,
    ]
    actual = pytlz.are_predicates_true(func, predicates)(x)
    assert actual == expected


@pytest.mark.parametrize("seed", [None, 0, random.Random(1), "invalid seed", 3.0])
def test_check_random_state(seed):
    if seed is None or isinstance(seed, (int, random.Random)):
        assert isinstance(pytlz.check_random_state(seed), random.Random)
    else:
        with pytest.raises(ValueError):
            # noinspection PyTypeChecker
            pytlz.check_random_state(seed)


@pytest.mark.parametrize("bias", [0.25, 0.5, 0.75, -0.1, 1.1, 11])
def test_coinflip(bias):
    if 0 <= bias <= 1:
        actual = {pytlz.coinflip(bias) for _ in range(30)}
        expected = {True, False}
        assert actual == expected
    else:
        with pytest.raises(ValueError):
            pytlz.coinflip(bias)


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
def test_collatz(n, expected):
    if n > 0:
        actual = tuple(pytlz.collatz(n))
        assert actual == expected
    else:
        with pytest.raises(ValueError):
            tuple(pytlz.collatz(n))


def test_concat_strings():
    strings = ["Hello", "World"]
    actual = pytlz.concat_strings(" ", strings)
    expected = " ".join(strings)
    assert actual == expected

    plus_concat = functools.partial(pytlz.concat_strings, " + ")
    actual = plus_concat(strings)
    expected = " + ".join(strings)
    assert actual == expected


def test_contrast_sets():
    x = {"a", "c", "b", "g", "h"}
    y = {"c", "d", "e", "f", "g"}
    summary = pytlz.contrast_sets(x, y)

    assert isinstance(summary, dict)
    assert summary["x"] == x
    assert summary["y"] == y
    assert summary["x | y"] == x.union(y)
    assert summary["x & y"] == x.intersection(y)
    assert summary["x - y"] == x.difference(y)
    assert summary["y - x"] == y.difference(x)
    assert summary["x ^ y"] == x.symmetric_difference(y)
    assert summary["jaccard"] == 0.25
    assert summary["overlap"] == 0.4
    assert summary["dice"] == 0.4
    assert summary["disjoint?"] is False
    assert summary["x == y"] is False
    assert summary["x <= y"] is False
    assert summary["x <  y"] is False
    assert summary["y <= x"] is False
    assert summary["y <  x"] is False

    lines = [
        "    x (n=5): {'a', 'b', 'c', ...}",
        "    y (n=5): {'c', 'd', 'e', ...}",
        "x | y (n=8): {'a', 'b', 'c', ...}",
        "x & y (n=2): {'c', 'g'}",
        "x - y (n=3): {'a', 'b', 'h'}",
        "y - x (n=3): {'d', 'e', 'f'}",
        "x ^ y (n=6): {'a', 'b', 'd', ...}",
        "jaccard = 0.25",
        "overlap = 0.4",
        "dice = 0.4",
        "disjoint?: False",
        "x == y: False",
        "x <= y: False",
        "x <  y: False",
        "y <= x: False",
        "y <  x: False",
    ]
    report = "\n".join(lines)
    assert summary["report"] == report


@pytest.mark.parametrize(
    "strings, expected",
    [
        (["path", "to", "file"], "path/to/file"),
        (["hdfs:", "path", "to", "file"], "hdfs:/path/to/file"),
        (["hdfs:/", "path", "to", "file"], "hdfs:/path/to/file"),
        (["hdfs://", "path", "to", "file"], "hdfs://path/to/file"),
    ],
)
def test_create_path(strings, expected):
    expected = expected.replace("/", os.sep)

    actual = pytlz.create_path(strings)
    assert actual == expected

    actual = pytlz.create_path(*strings)
    assert actual == expected


@pytest.mark.parametrize(
    "d, expected",
    [
        (dt.date(2022, 1, 1), "2022-01-01"),
        (dt.date(2022, 1, 31), "2022-01-31"),
    ],
)
def test_date_to_str(d, expected):
    actual = pytlz.date_to_str(d)
    assert actual == expected


@pytest.mark.parametrize(
    "xmin, xmax, factor, expected",
    [
        (0, 1, -0.1, None),
        (0, 1, 0.0, (0, 1)),
        (0, 1, 0.05, (-0.05, 1.05)),
        (0, 1, 0.1, (-0.1, 1.1)),
        (-1, 10, 0.1, (-2.1, 11.1)),
        (0, 10, 0.1, (-1.0, 11.0)),
        (1, 0, -0.1, None),
        (1, 0, 0.0, (0, 1)),
        (1, 0, 0.05, (-0.05, 1.05)),
    ],
)
def test_extend_range(xmin, xmax, factor, expected):
    extend_range = functools.partial(pytlz.extend_range, factor=factor)
    if factor >= 0:
        actual = extend_range(xmin, xmax)
        assert actual == expected
    else:
        with pytest.raises(ValueError):
            extend_range(xmin, xmax)


def test_fibonacci():
    actual = tuple(toolz.take(16, pytlz.fibonacci()))
    expected = (0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610)
    assert actual == expected


@pytest.mark.parametrize(
    "items, expected",
    [
        ([1, 2, 3], [1, 2, 3]),
        ((1, 2, 3), [1, 2, 3]),
        (range(1, 7), [1, 2, 3, 4, 5, 6]),
        ([[1, 2], [3, 4], [5, 6]], [1, 2, 3, 4, 5, 6]),
        ([[], [1, 2], [3, 4, 5], [6]], [1, 2, 3, 4, 5, 6]),
        ([[1, 2], 3, (4, 5), (6,)], [1, 2, 3, 4, 5, 6]),
        ([[[1, 2]], [3], 4, [[[5]]], [[[[6]]]]], [1, 2, 3, 4, 5, 6]),
        ([1, [2, 3, 4], [[5, 6]]], [1, 2, 3, 4, 5, 6]),
        ([1, [2, 3, 4, 5], 6, []], [1, 2, 3, 4, 5, 6]),
        ([[1, 2], [3, 4, 5], 6], [1, 2, 3, 4, 5, 6]),
        (([[1, 2], [3, 4, 5], 6],), [1, 2, 3, 4, 5, 6]),
        ([iter([1, (2, 3)]), 4, [], iter([[[5]], 6])], [1, 2, 3, 4, 5, 6]),
        (
            [["one", 2], 3, [4, "five"], ["six"]],
            ["one", 2, 3, 4, "five", "six"],
        ),
        (map(lambda x: 2 * x, range(1, 7)), [2, 4, 6, 8, 10, 12]),
        ((2 * x for x in range(1, 7)), [2, 4, 6, 8, 10, 12]),
        (tuple(2 * x for x in range(1, 7)), [2, 4, 6, 8, 10, 12]),
        (list(2 * x for x in range(1, 7)), [2, 4, 6, 8, 10, 12]),
        (([-1], 0, range(1, 7)), [-1, 0, 1, 2, 3, 4, 5, 6]),
        (([-1], 0, map(lambda x: 2 * x, range(1, 4))), [-1, 0, 2, 4, 6]),
        (([-1], 0, (2 * x for x in range(1, 4))), [-1, 0, 2, 4, 6]),
        (([-1], 0, tuple(2 * x for x in range(1, 4))), [-1, 0, 2, 4, 6]),
        (([-1], 0, list(2 * x for x in range(1, 4))), [-1, 0, 2, 4, 6]),
        (filter(lambda x: x % 2 == 0, range(1, 7)), [2, 4, 6]),
        ((-1, filter(lambda x: x % 2 == 0, range(1, 7))), [-1, 2, 4, 6]),
        (([-1], filter(lambda x: x % 2 == 0, range(1, 7))), [-1, 2, 4, 6]),
    ],
)
def test_flatten(items, expected):
    actual = list(pytlz.flatten(items))
    assert actual == expected


def test_func_name():
    def foobar():
        return pytlz.func_name()

    actual = foobar()
    expected = "foobar"
    assert actual == expected


@pytest.mark.parametrize("x", [-1, 0, 1, 2, 3, 3.14, 4, 5, 6, 7, 8, 9, 10, 11.0])
def test_isdivisibleby(x):
    actual = pytlz.isdivisibleby(2, x)
    expected = x % 2 == 0
    assert actual == expected


@pytest.mark.parametrize("x", [-1, 0, 1, 2, 3, 3.14, 4, 5, 6, 7, 8, 9, 10, 11.0])
def test_iseven(x):
    actual = pytlz.iseven(x)
    expected = x % 2 == 0
    assert actual == expected


@pytest.mark.parametrize("x", [-1, 0, 1, 2, 3, 3.14, 4, 5, 6, 7, 8, 9, 10, 11.0])
def test_isodd(x):
    actual = pytlz.isodd(x)
    is_even_number = x % 2 == 0
    expected = not is_even_number
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
    actual = pytlz.num_to_str(n)
    assert actual == expected


@pytest.mark.parametrize(
    "func, expected",
    [
        (set.intersection, {2}),
        (set.union, {0, 1, 2, 3, 4, 6, 8}),
        (set.difference, {0, 1, 3}),
        (set.symmetric_difference, {0, 1, 2, 3, 4, 8}),
    ],
)
def test_reduce_sets(func, expected):
    x = {0, 1, 2, 3}
    y = {2, 4, 6}
    z = {2, 6, 8}

    f = pytlz.reduce_sets(func)
    assert isinstance(f, toolz.curry)

    actual = f(x, y, z)
    assert isinstance(actual, set)
    assert actual == expected

    actual = pytlz.reduce_sets(func, [x, y, z])
    assert isinstance(actual, set)
    assert actual == expected


@pytest.mark.parametrize(
    "x, n, expected",
    [
        (math.inf, 3, math.inf),
        (1234, 0, None),
        (0, 3, 0),
        (0.0, 3, 0.0),
        (1e-12, 3, 1e-12),
        (987654321.123456789, 1, 1000000000.0),
        (987654321.123456789, 2, 990000000.0),
        (987654321.123456789, 3, 988000000.0),
        (987654321.123456789, 4, 987700000.0),
        (987654321.123456789, 5, 987650000.0),
        (987654321.123456789, 6, 987654000.0),
        (987654321.123456789, 7, 987654300.0),
        (987654321.123456789, 8, 987654320.0),
        (987654321.123456789, 9, 987654321.0),
        (987654321.123456789, 10, 987654321.1),
        (987654321.123456789, 11, 987654321.12),
        (987654321.123456789, 12, 987654321.123),
        (987654321.123456789, 13, 987654321.1235),
        (1.123456789, 1, 1.0),
        (1.123456789, 2, 1.1),
        (1.123456789, 3, 1.12),
        (1.123456789, 4, 1.123),
        (0.123456789, 1, 0.1),
        (0.123456789, 2, 0.12),
        (0.123456789, 3, 0.123),
        (0.123456789, 4, 0.1235),
        (1234, 1, 1000),
        (1234, 2, 1200),
        (1234, 3, 1230),
        (1234, 4, 1234),
        (-1.4142135623730951, 1, -1.0),
        (-1.4142135623730951, 2, -1.4),
        (-1.4142135623730951, 3, -1.41),
        (-1.4142135623730951, 4, -1.414),
        (14393237.76, 1, 10000000.0),
        (14393237.76, 2, 14000000.0),
        (14393237.76, 3, 14400000.0),
        (14393237.76, 4, 14390000.0),
    ],
)
def test_signif(x, n, expected):
    f = functools.partial(pytlz.signif, n=n)
    if n > 0:
        actual = f(x)
        assert actual == expected
    else:
        with pytest.raises(ValueError):
            f(x)


def test_source_code():
    def greet():
        return "Hello, World!"

    actual = pytlz.source_code(greet)
    expected = '    def greet():\n        return "Hello, World!"\n'
    assert actual == expected


class TestRegexFunctions:
    def test_filter_regex(self, zen_of_python: Tuple[str]):
        actual = list(pytlz.filter_regex("python", zen_of_python))
        expected = ["The Zen of Python, by Tim Peters"]
        assert actual == expected, "filter_python_regex fails"

        filter_regex__better = functools.partial(pytlz.filter_regex, "better")
        actual = list(filter_regex__better(*zen_of_python))
        expected = [
            "Beautiful is better than ugly.",
            "Explicit is better than implicit.",
            "Simple is better than complex.",
            "Complex is better than complicated.",
            "Flat is better than nested.",
            "Sparse is better than dense.",
            "Now is better than never.",
            "Although never is often better than *right* now.",
        ]
        assert actual == expected, "filter_better_regex fails"

    def test_map_regex(self, zen_of_python: Tuple[str]):
        actual = list(pytlz.map_regex("python", zen_of_python))
        expected = [["Python"]] + [[] for _ in range(19)]
        assert actual == expected

        filter_regex__better = functools.partial(pytlz.map_regex, "better")
        actual = list(filter_regex__better(zen_of_python))
        expected = (
            [[]]
            + [["better"] for _ in range(6)]
            + [[] for _ in range(8)]
            + [["better"] for _ in range(2)]
            + [[] for _ in range(3)]
        )
        assert actual == expected

    @pytest.fixture(scope="class")
    def zen_of_python(self) -> Tuple[str]:
        return (
            "The Zen of Python, by Tim Peters",
            "Beautiful is better than ugly.",
            "Explicit is better than implicit.",
            "Simple is better than complex.",
            "Complex is better than complicated.",
            "Flat is better than nested.",
            "Sparse is better than dense.",
            "Readability counts.",
            "Special cases aren't special enough to break the rules.",
            "Although practicality beats purity.",
            "Errors should never pass silently.",
            "Unless explicitly silenced.",
            "In the face of ambiguity, refuse the temptation to guess.",
            "There should be one--and preferably only one--obvious way to do it.",
            "Although that way may not be obvious at first unless you're Dutch.",
            "Now is better than never.",
            "Although never is often better than *right* now.",
            "If the implementation is hard to explain, it's a bad idea.",
            "If the implementation is easy to explain, it may be a good idea.",
            "Namespaces are one honking great idea -- let's do more of those!",
        )
