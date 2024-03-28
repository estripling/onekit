import datetime as dt
import functools
import math
import operator
import os
import random
import re
import time
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import pytest
import toolz
from toolz import curried

import onekit.pythonkit as pk


@pytest.mark.parametrize("kind", ["zip", "gztar"])
def test_archive_files(kind: str):
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir).joinpath("test_file_for_archive_files.txt")
        glue_strings = functools.partial(pk.concat_strings, "")

        with path.open("w") as fh:
            fh.write(glue_strings("Hello, World!", os.linesep))

        dir_path = Path(tmpdir).joinpath("test_directory_for_archive_files/")
        os.makedirs(dir_path)
        path2 = Path(dir_path).joinpath("test_file_for_archive_files2.txt")

        with path2.open("w") as fh:
            fh.write(glue_strings("Hello, Again!", os.linesep))

        pk.archive_files(tmpdir, name="archive", kind=kind)

    if kind == "gztar":
        kind = "tar.gz"

    os.remove(f"archive.{kind}")


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
def test_are_predicates_true(x: int, func: Callable, expected: bool):
    predicates = [
        lambda x: x % 3 == 0,
        lambda x: x % 5 == 0,
    ]
    actual = pk.are_predicates_true(func, predicates)(x)
    assert actual == expected


@pytest.mark.parametrize("seed", [None, 0, random.Random(1), "invalid seed", 3.0])
def test_check_random_state(seed):
    if seed is None or isinstance(seed, (int, random.Random)):
        assert isinstance(pk.check_random_state(seed), random.Random)
    else:
        with pytest.raises(ValueError):
            # noinspection PyTypeChecker
            pk.check_random_state(seed)


@pytest.mark.parametrize("bias", [0.25, 0.5, 0.75, -0.1, 1.1, 11])
def test_coinflip(bias: float):
    if 0 <= bias <= 1:
        actual = {pk.coinflip(bias) for _ in range(30)}
        expected = {True, False}
        assert actual == expected
    else:
        with pytest.raises(ValueError):
            pk.coinflip(bias)


def test_concat_strings():
    strings = ["Hello", "World"]
    actual = pk.concat_strings(" ", strings)
    expected = " ".join(strings)
    assert actual == expected

    plus_concat = functools.partial(pk.concat_strings, " + ")
    actual = plus_concat(strings)
    expected = " + ".join(strings)
    assert actual == expected


def test_contrast_sets():
    x = {"a", "c", "b", "g", "h"}
    y = {"c", "d", "e", "f", "g"}
    summary = pk.contrast_sets(x, y)

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
def test_create_path(strings: List[str], expected: str):
    expected = expected.replace("/", os.sep)

    actual = pk.create_path(strings)
    assert actual == expected

    actual = pk.create_path(*strings)
    assert actual == expected


@pytest.mark.parametrize(
    "d, expected",
    [
        (dt.date(2022, 1, 1), "2022-01-01"),
        (dt.date(2022, 1, 31), "2022-01-31"),
    ],
)
def test_date_to_str(d: dt.date, expected: str):
    actual = pk.date_to_str(d)
    assert actual == expected


def test_datecount():
    d0 = dt.date(2022, 1, 1)

    actual = toolz.pipe(pk.datecount(d0, forward=True), curried.take(3), list)
    expected = [dt.date(2022, 1, 1), dt.date(2022, 1, 2), dt.date(2022, 1, 3)]
    assert actual == expected

    actual = toolz.pipe(pk.datecount(d0, forward=False), curried.take(3), list)
    expected = [dt.date(2022, 1, 1), dt.date(2021, 12, 31), dt.date(2021, 12, 30)]
    assert actual == expected


@pytest.mark.parametrize(
    "start, end, expected",
    [
        (
            dt.date(2022, 1, 1),
            dt.date(2022, 1, 5),
            (
                dt.date(2022, 1, 1),
                dt.date(2022, 1, 2),
                dt.date(2022, 1, 3),
                dt.date(2022, 1, 4),
                dt.date(2022, 1, 5),
            ),
        ),
        (
            dt.date(2022, 1, 5),
            dt.date(2022, 1, 1),
            (
                dt.date(2022, 1, 1),
                dt.date(2022, 1, 2),
                dt.date(2022, 1, 3),
                dt.date(2022, 1, 4),
                dt.date(2022, 1, 5),
            ),
        ),
        (dt.date(2022, 1, 1), dt.date(2022, 1, 1), (dt.date(2022, 1, 1),)),
    ],
)
def test_daterange(start: dt.date, end: dt.date, expected: Tuple[dt.date]):
    actual = tuple(pk.daterange(start, end))
    assert actual == expected

    actual = tuple(pk.daterange(start, end, incl_min=False))
    assert actual == expected[1:]

    actual = tuple(pk.daterange(start, end, incl_max=False))
    assert actual == expected[:-1]

    actual = tuple(pk.daterange(start, end, incl_min=False, incl_max=False))
    assert actual == expected[1:-1]


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
def test_extend_range(
    xmin: float,
    xmax: float,
    factor: float,
    expected: Tuple[float, float],
):
    extend_range = functools.partial(pk.extend_range, factor=factor)
    if factor >= 0:
        actual = extend_range(xmin, xmax)
        assert actual == expected
    else:
        with pytest.raises(ValueError):
            extend_range(xmin, xmax)


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
def test_flatten(items: Any, expected: List[Any]):
    actual = list(pk.flatten(items))
    assert actual == expected


def test_func_name():
    def foobar():
        return pk.func_name()

    actual = foobar()
    expected = "foobar"
    assert actual == expected


def test_headline():
    actual = pk.headline("Hello, World!", n=30)
    expected = "------- Hello, World! --------"
    assert actual == expected


@pytest.mark.parametrize(
    "lft_str, rgt_str, expected",
    [
        ("hello", "hello", "hello\n     \nhello"),
        ("hello", "hallo", "hello\n |   \nhallo"),
        ("hello", "hell", "hello\n    |\nhell"),
        ("hello", "hall", "hello\n |  |\nhall"),
        ("hall", "hello", "hall\n |  |\nhello"),
        ("1234", "11342", "1234\n |  |\n11342"),
        ("1234.56", "1234,56", "1234.56\n    |  \n1234,56"),
    ],
)
def test_highlight_string_differences(lft_str: str, rgt_str: str, expected: str):
    actual = pk.highlight_string_differences(lft_str, rgt_str)
    assert actual == expected


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (-2.0, None),
        (-1, None),
        (0, "0s"),
        (1, "1s"),
        (59, "59s"),
        (59.0, "59s"),
        (60, "1m"),
        (60.1, "1m"),
        (61, "1m 1s"),
        (61.1, "1m 1s"),
        (120, "2m"),
        (120.1, "2m"),
        (60 * 60, "1h"),
        (60 * 60 + 1, "1h 1s"),
        (60 * 60 * 24, "1d"),
        (60 * 60 * 24 + 1, "1d 1s"),
        (110.0, "1m 50s"),
        (0.4142135623730951, "0.414214s"),
        (0.5, "0.5s"),
        (1.4142135623730951, "1.41421s"),
        (1.5, "1.5s"),
        (2.4142135623730951, "2.41421s"),
        (59.4142135623730951, "59.4142s"),
        (60.4142135623730951, "1m"),
        (60.5142135623730951, "1m 1s"),
        (60 * 60 * 24 + 123456, "2d 10h 17m 36s"),
    ],
)
def test_humantime(seconds: Union[int, float], expected: Optional[str]):
    if seconds >= 0:
        actual = pk.humantime(seconds)
        assert actual == expected
    else:
        with pytest.raises(ValueError):
            pk.humantime(seconds)


@pytest.mark.parametrize(
    "d, expected",
    [
        (dt.date(2022, 1, 1), dt.date(2022, 1, 31)),
        (dt.date(2022, 2, 1), dt.date(2022, 2, 28)),
        (dt.date(2022, 3, 1), dt.date(2022, 3, 31)),
        (dt.date(2022, 4, 1), dt.date(2022, 4, 30)),
        (dt.date(2022, 5, 1), dt.date(2022, 5, 31)),
        (dt.date(2022, 6, 1), dt.date(2022, 6, 30)),
        (dt.date(2022, 7, 1), dt.date(2022, 7, 31)),
        (dt.date(2022, 8, 1), dt.date(2022, 8, 31)),
        (dt.date(2022, 9, 1), dt.date(2022, 9, 30)),
        (dt.date(2022, 10, 1), dt.date(2022, 10, 31)),
        (dt.date(2022, 11, 1), dt.date(2022, 11, 30)),
        (dt.date(2022, 12, 1), dt.date(2022, 12, 31)),
        (dt.date(1970, 1, 1), dt.date(1970, 1, 31)),
        (dt.date(1970, 1, 15), dt.date(1970, 1, 31)),
        (dt.date(1970, 1, 31), dt.date(1970, 1, 31)),
        (dt.date(2020, 2, 2), dt.date(2020, 2, 29)),
        (dt.date(2022, 2, 3), dt.date(2022, 2, 28)),
        (dt.date(2000, 2, 4), dt.date(2000, 2, 29)),
        (dt.date(1900, 2, 5), dt.date(1900, 2, 28)),
        (dt.date(2012, 2, 27), dt.date(2012, 2, 29)),
        (dt.date(2012, 2, 28), dt.date(2012, 2, 29)),
        (dt.date(2012, 2, 29), dt.date(2012, 2, 29)),
    ],
)
def test_last_date_of_month(d: dt.date, expected: dt.date):
    actual = pk.last_date_of_month(d.year, d.month)
    assert actual == expected


def test_lazy_read_lines():
    expected = ("one", "two", "three")

    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir).joinpath("test_file_for_lazy_read_lines.txt")

        with path.open("w") as fh:
            fh.write(pk.concat_strings(os.linesep, expected))

        actual = toolz.pipe(pk.lazy_read_lines(path), curried.map(str.rstrip), tuple)
        assert actual == expected

        for i, line in enumerate(pk.lazy_read_lines(str(path))):
            actual = line.replace(os.linesep, "")
            assert actual == expected[i]

        with pytest.raises(FileNotFoundError):
            tuple(pk.lazy_read_lines(Path(tmpdir).joinpath("./not_exist.txt")))


@pytest.mark.parametrize(
    "x, expected",
    [
        (1, "1"),
        (10, "10"),
        (100, "100"),
        (1000, "1_000"),
        (1000000, "1_000_000"),
        (100000.0, "100_000.0"),
    ],
)
def test_num_to_str(x: Union[int, float], expected: str):
    actual = pk.num_to_str(x)
    assert actual == expected


@pytest.mark.parametrize(
    "d1, d2, expected",
    [
        (dt.date(2022, 8, 1), dt.date(2022, 8, 1), 1),
        (dt.date(2022, 8, 1), dt.date(2022, 8, 7), 7),
        (dt.date(2022, 8, 7), dt.date(2022, 8, 1), 7),
        (dt.date(2014, 1, 1), dt.date(2016, 5, 6), 857),
    ],
)
def test_number_of_days(d1: dt.date, d2: dt.date, expected: int):
    actual = pk.number_of_days(d1, d2)
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
def test_reduce_sets(func: Callable, expected: Set[int]):
    x = {0, 1, 2, 3}
    y = {2, 4, 6}
    z = {2, 6, 8}

    f = pk.reduce_sets(func)
    assert isinstance(f, toolz.curry)

    actual = f(x, y, z)
    assert isinstance(actual, set)
    assert actual == expected

    actual = pk.reduce_sets(func, [x, y, z])
    assert isinstance(actual, set)
    assert actual == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("no punctuation", "no punctuation"),
        (" string with whitespaces ", " string with whitespaces "),
        ("CAPITAL LETTERS", "CAPITAL LETTERS"),
        ("exclamation mark!", "exclamation mark"),
        ('quotation mark"', "quotation mark"),
        ("hash#", "hash"),
        ("dollar$", "dollar"),
        ("percentage%", "percentage"),
        ("ampersand&", "ampersand"),
        ("apostrophe'", "apostrophe"),
        ("asterisk*", "asterisk"),
        ("plus+", "plus"),
        ("comma,", "comma"),
        ("dash-", "dash"),
        ("period.", "period"),
        ("slash/", "slash"),
        ("colon:", "colon"),
        ("semicolon;", "semicolon"),
        ("less than sign<", "less than sign"),
        ("equal sign=", "equal sign"),
        ("greater than sign>", "greater than sign"),
        ("question mark?", "question mark"),
        ("at sign@", "at sign"),
        ("backslash\\", "backslash"),
        ("caret^", "caret"),
        ("underscore_", "underscore"),
        ("backtick`", "backtick"),
        ("vertical bar symbol|", "vertical bar symbol"),
        ("tilde~", "tilde"),
        ("(round brackets)", "round brackets"),
        ("{curly brackets}", "curly brackets"),
        ("[square brackets]", "square brackets"),
    ],
)
def test_remove_punctuation(text: str, expected: str):
    actual = pk.remove_punctuation(text)
    assert actual == expected


@pytest.mark.parametrize(
    "n, d0, expected",
    [
        (0, dt.date(2022, 1, 1), dt.date(2022, 1, 1)),
        (1, dt.date(2022, 1, 1), dt.date(2022, 1, 2)),
        (-1, dt.date(2022, 1, 1), dt.date(2021, 12, 31)),
        (2, dt.date(2022, 1, 1), dt.date(2022, 1, 3)),
        (-2, dt.date(2022, 1, 1), dt.date(2021, 12, 30)),
        (3, dt.date(2022, 1, 1), dt.date(2022, 1, 4)),
        (-3, dt.date(2022, 1, 1), dt.date(2021, 12, 29)),
        (7, dt.date(2022, 8, 1), dt.date(2022, 8, 8)),
        (-7, dt.date(2022, 8, 8), dt.date(2022, 8, 1)),
        (30, dt.date(2022, 8, 1), dt.date(2022, 8, 31)),
        (27, dt.date(2022, 2, 1), dt.date(2022, 2, 28)),
        (28, dt.date(2022, 2, 1), dt.date(2022, 3, 1)),
        (27, dt.date(2020, 2, 1), dt.date(2020, 2, 28)),
        (28, dt.date(2020, 2, 1), dt.date(2020, 2, 29)),
        (29, dt.date(2020, 2, 1), dt.date(2020, 3, 1)),
    ],
)
def test_relative_date(n: int, d0: dt.date, expected: dt.date):
    actual = pk.relative_date(d0, n)
    assert actual == expected

    dates = pk.daterange(d0, actual, incl_min=True, incl_max=True)
    n_days = curried.count(dates)
    n_days_expected = abs(n) + 1
    assert n_days == n_days_expected


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
def test_signif(x: Union[int, float], n: int, expected: Union[int, float]):
    f = functools.partial(pk.signif, n=n)
    if n > 0:
        actual = f(x)
        assert actual == expected
    else:
        with pytest.raises(ValueError):
            f(x)


def test_source_code():
    def greet():
        return "Hello, World!"

    actual = pk.source_code(greet)
    expected = '    def greet():\n        return "Hello, World!"\n'
    assert actual == expected


class TestOp:
    @pytest.mark.parametrize("x, expected", [(0, 1), (1, 2), (10, 11), (21, 22)])
    def test_add(self, x: Any, expected: Any):
        inc = pk.op(operator.add, 1)
        actual = inc(x)
        assert actual == expected

    @pytest.mark.parametrize("x, expected", [(0, -1), (1, 0), (10, 9), (21, 20)])
    def test_sub(self, x: Any, expected: Any):
        dec = pk.op(operator.sub, 1)
        actual = dec(x)
        assert actual == expected

    @pytest.mark.parametrize("x, expected", [(0, 0), (1, 2), (10, 20), (21, 42)])
    def test_mul(self, x: Any, expected: Any):
        mul2 = pk.op(operator.mul, 2)
        actual = mul2(x)
        assert actual == expected

    @pytest.mark.parametrize("x, expected", [(0, 0), (1, 0.5), (10, 5), (21, 10.5)])
    def test_truediv(self, x: Any, expected: Any):
        div2 = pk.op(operator.truediv, 2)
        actual = div2(x)
        assert actual == expected


class TestPromptYesNo:
    @pytest.mark.parametrize(
        "default, answer, expected",
        [
            (None, "yes", True),
            (None, "no", False),
            ("yes", "yes", True),
            ("yes", "no", False),
            ("no", "yes", True),
            ("no", "no", False),
            ("yes", "\n", True),
            ("no", "\n", False),
        ],
    )
    def test_default_call(
        self,
        monkeypatch,
        default: Optional[str],
        answer: str,
        expected: bool,
    ):
        monkeypatch.setattr("sys.stdin", StringIO(answer))
        actual = pk.prompt_yes_no("Do you like onekit?", default=default)
        assert actual == expected

    @pytest.mark.parametrize("default", [1, "noo", "yeah"])
    def test_invalid_default_value(self, default: Optional[str]):
        with pytest.raises(ValueError):
            pk.prompt_yes_no("Do you like onekit?", default=default)

    def test_subsequent_prompt(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", StringIO("yay"))
        with pytest.raises(EOFError):
            pk.prompt_yes_no("Do you like onekit?", default="yes")


class TestRegexFunctions:
    def test_filter_regex(self, zen_of_python: Tuple[str]):
        actual = list(pk.filter_regex("python", zen_of_python))
        expected = ["The Zen of Python, by Tim Peters"]
        assert actual == expected

        filter_regex__better = functools.partial(pk.filter_regex, "better")
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
        assert actual == expected

    def test_map_regex(self, zen_of_python: Tuple[str]):
        actual = list(pk.map_regex("python", zen_of_python))
        expected = [["Python"]] + [[] for _ in range(19)]
        assert actual == expected

        filter_regex__better = functools.partial(pk.map_regex, "better")
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


class TestStopwatch:
    def test_context_manager__default_call(
        self,
        slumber,
        regex_default_message,
        capsys,
    ):
        with pk.stopwatch():
            slumber()

        actual = capsys.readouterr().out
        expected = regex_default_message
        assert re.search(expected, actual) is not None

    def test_context_manager__instance(self, slumber, regex_default_message):
        with pk.stopwatch() as sw:
            slumber()

        actual = str(sw)
        expected = regex_default_message
        assert re.search(expected, actual) is not None

    @pytest.mark.parametrize("label", [None, "lbl", 1])
    def test_context_manager__label(
        self,
        slumber,
        regex_default_message,
        label,
    ):
        with pk.stopwatch(label) as sw:
            slumber()

        actual = str(sw)
        expected = (
            regex_default_message
            if label is None
            else regex_default_message.replace("$", f" - {label}$")
        )
        assert re.search(expected, actual) is not None
        assert sw.label is None if label is None else sw.label == label

        with pytest.raises(AttributeError, match=r"can't set attribute"):
            sw.label = label

        with pytest.raises(
            TypeError,
            match=r"got some positional-only arguments passed as keyword arguments",
        ):
            with pk.stopwatch(label=label):
                slumber()

    @pytest.mark.parametrize("flush", [True, False])
    def test_context_manager__flush(self, slumber, regex_default_message, flush):
        with pk.stopwatch(flush=flush) as sw:
            slumber()

        actual = str(sw)
        expected = regex_default_message
        assert re.search(expected, actual) is not None
        assert sw.flush == flush

        with pytest.raises(AttributeError, match=r"can't set attribute"):
            sw.flush = flush

    @pytest.mark.parametrize(
        "case,fmt",
        [
            (1, None),
            (2, "%Y-%m-%d %H:%M:%S"),
            (3, "%H:%M:%S"),
            (4, "%A, %d %B %Y %H:%M:%S"),
        ],
    )
    def test_context_manager__fmt(
        self,
        slumber,
        regex_default_message,
        case,
        fmt,
        default_fmt="%Y-%m-%d %H:%M:%S",
    ):
        with pk.stopwatch(fmt=fmt) as sw:
            slumber()

        actual = str(sw)
        expected = (
            regex_default_message
            if case in (1, 2)
            else self.create_regex_for_message(r"\d{2}:\d{2}:\d{2}")
            if case == 3
            else self.create_regex_for_message(
                r"\w+, \d{2} \w+ \d{4} \d{2}:\d{2}:\d{2}"
            )
            if case == 4
            else None
        )
        assert re.search(expected, actual) is not None
        assert sw.fmt == default_fmt if fmt is None else sw.fmt == fmt

        # change timestamp format but not data
        sw.fmt = default_fmt

        with pytest.raises(AttributeError, match=r"can't set attribute"):
            sw.start_time = dt.datetime.now()

        with pytest.raises(AttributeError, match=r"can't set attribute"):
            sw.stop_time = dt.datetime.now()

        with pytest.raises(AttributeError, match=r"can't set attribute"):
            sw.elapsed_time = dt.timedelta(days=42)

        actual = str(sw)
        expected = regex_default_message
        assert re.search(expected, actual) is not None

    @pytest.mark.parametrize("label", [None, "lbl"])
    @pytest.mark.parametrize("flush", [True, False])
    @pytest.mark.parametrize(
        "case,fmt", [(1, None), (2, "%Y-%m-%d %H:%M:%S"), (3, "%H:%M:%S")]
    )
    def test_context_manager__many_param(
        self,
        slumber,
        regex_default_message,
        label,
        flush,
        case,
        fmt,
        default_fmt="%Y-%m-%d %H:%M:%S",
    ):
        with pk.stopwatch(label, flush=flush, fmt=fmt) as sw:
            slumber()

        actual = str(sw)
        expected_message = (
            regex_default_message
            if case in (1, 2)
            else self.create_regex_for_message(r"\d{2}:\d{2}:\d{2}")
            if case == 3
            else None
        )
        expected = (
            expected_message
            if label is None
            else expected_message.replace("$", f" - {label}$")
        )
        assert re.search(expected, actual) is not None
        assert sw.label == label
        assert sw.flush == flush
        assert sw.fmt == default_fmt if fmt is None else sw.fmt == fmt

        with pytest.raises(AttributeError, match=r"can't set attribute"):
            sw.label = label

        with pytest.raises(AttributeError, match=r"can't set attribute"):
            sw.flush = flush

        with pytest.raises(AttributeError, match=r"can't set attribute"):
            sw.start_time = dt.datetime.now()

        with pytest.raises(AttributeError, match=r"can't set attribute"):
            sw.stop_time = dt.datetime.now()

        with pytest.raises(AttributeError, match=r"can't set attribute"):
            sw.elapsed_time = dt.timedelta(days=42)

    def test_context_manager__total_elapsed_time(self, slumber, regex_default_message):
        with pk.stopwatch(1) as sw1:
            slumber()

        with pk.stopwatch(2) as sw2:
            slumber()

        with pk.stopwatch(3) as sw3:
            slumber()

        for i, sw in enumerate([sw1, sw2, sw3]):
            label = str(i + 1)
            actual = str(sw)
            expected = regex_default_message.replace("$", f" - {label}$")
            assert re.search(expected, actual) is not None

        additions = [
            (1, sum([sw1])),
            (2, sw1 + sw2),
            (3, sum([sw2], start=sw1)),
            (4, sum([sw1, sw2])),
            (5, sw1 + sw2 + sw3),
            (6, sum([sw2, sw3], start=sw1)),
            (7, sum([sw1, sw2, sw3])),
        ]
        for case, total in additions:
            actual = str(total)
            n_stopwatches = 1 if case == 1 else 2 if 2 <= case <= 4 else 3
            expected = rf"^0\.0{n_stopwatches}(\d*)?s - total elapsed time$"
            assert re.search(expected, actual) is not None

    def test_decorator__default_call(self, slumber, regex_default_message, capsys):
        @pk.stopwatch()
        def func():
            slumber()

        func()

        actual = capsys.readouterr().out
        expected = regex_default_message.replace("$", f" - {func.__name__}$")
        assert re.search(expected, actual) is not None

    def test_decorator__label(
        self,
        slumber,
        regex_default_message,
        capsys,
        label="lbl",
    ):
        @pk.stopwatch(label)
        def func():
            slumber()

        func()

        actual = capsys.readouterr().out
        expected = regex_default_message.replace("$", f" - {label}$")
        assert re.search(expected, actual) is not None

    @pytest.mark.parametrize("flush", [True, False])
    def test_decorator__flush(self, slumber, regex_default_message, flush, capsys):
        @pk.stopwatch(flush=flush)
        def func():
            slumber()

        func()

        actual = capsys.readouterr().out
        expected = regex_default_message.replace("$", f" - {func.__name__}$")
        assert re.search(expected, actual) is not None

    @pytest.mark.parametrize(
        "case,fmt",
        [
            (1, None),
            (2, "%Y-%m-%d %H:%M:%S"),
            (3, "%H:%M:%S"),
            (4, "%A, %d %B %Y %H:%M:%S"),
        ],
    )
    def test_decorator__fmt(
        self,
        slumber,
        regex_default_message,
        case,
        fmt,
        capsys,
    ):
        @pk.stopwatch(fmt=fmt)
        def func():
            slumber()

        func()

        actual = capsys.readouterr().out
        expected_message = (
            regex_default_message
            if case in [1, 2]
            else self.create_regex_for_message(r"\d{2}:\d{2}:\d{2}")
            if case == 3
            else self.create_regex_for_message(
                r"\w+, \d{2} \w+ \d{4} \d{2}:\d{2}:\d{2}"
            )
            if case == 4
            else None
        )
        expected = expected_message.replace("$", f" - {func.__name__}$")
        assert re.search(expected, actual) is not None

    @pytest.mark.parametrize("label", [None, "lbl"])
    @pytest.mark.parametrize("flush", [True, False])
    @pytest.mark.parametrize(
        "case,fmt", [(1, None), (2, "%Y-%m-%d %H:%M:%S"), (3, "%H:%M:%S")]
    )
    def test_decorator__many_param(
        self,
        slumber,
        regex_default_message,
        label,
        flush,
        case,
        fmt,
        capsys,
    ):
        @pk.stopwatch(label, fmt=fmt, flush=flush)
        def func():
            slumber()

        func()

        actual = capsys.readouterr().out
        expected_message = (
            regex_default_message
            if case in (1, 2)
            else self.create_regex_for_message(r"\d{2}:\d{2}:\d{2}")
            if case == 3
            else None
        )
        expected = (
            expected_message.replace("$", f" - {func.__name__}$")
            if label is None
            else expected_message.replace("$", f" - {label}$")
        )
        assert re.search(expected, actual) is not None

    @pytest.mark.parametrize("label", [True, 0.0, set(), [2]])
    def test_raises_type_error__label(self, slumber, label):
        with pytest.raises(
            TypeError,
            match=r"label=.* - must be str, int, or NoneType",
        ):
            with pk.stopwatch(label):
                slumber()

    @pytest.mark.parametrize("flush", [None, 0, 1.0, set(), [2]])
    def test_raises_type_error__flush(self, slumber, flush):
        with pytest.raises(TypeError, match=r"flush=.* - must be bool"):
            with pk.stopwatch(flush=flush):
                slumber()

    @pytest.mark.parametrize("fmt", [True, 0, 1.0, set(), [2]])
    def test_raises_type_error__fmt(self, slumber, fmt):
        with pytest.raises(TypeError, match=r"fmt=.* - must be str or NoneType"):
            with pk.stopwatch(fmt=fmt):
                slumber()

    @pytest.mark.parametrize("fmt", [True, 0, 1.0, set(), [2]])
    def test_raises_type_error__fmt_setter(self, slumber, fmt):
        with pk.stopwatch() as sw:
            slumber()

        with pytest.raises(TypeError, match=r"value=.* - `fmt` must be str"):
            sw.fmt = fmt

    @pytest.fixture(scope="class")
    def slumber(self):
        def _():
            time.sleep(0.01)

        return _

    @pytest.fixture(scope="class")
    def regex_default_message(self, regex_default_fmt):
        """Regex: default output message."""
        return self.create_regex_for_message(regex_default_fmt)

    @pytest.fixture(scope="class")
    def regex_default_fmt(self):
        """Regex: default timestamp format."""
        return r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"

    @staticmethod
    def create_regex_for_message(regex_fmt):
        return rf"^{regex_fmt} -> {regex_fmt} = 0\.01(\d*)?s$"


@pytest.mark.parametrize(
    "string, expected",
    [
        ("2022-01-01", dt.date(2022, 1, 1)),
        ("2022-01-31", dt.date(2022, 1, 31)),
    ],
)
def test_str_to_date(string: str, expected: dt.date):
    actual = pk.str_to_date(string)
    assert actual == expected


@pytest.mark.parametrize(
    "d, expected",
    [
        (dt.date(2022, 8, 1), "Mon"),
        (dt.date(2022, 8, 2), "Tue"),
        (dt.date(2022, 8, 3), "Wed"),
        (dt.date(2022, 8, 4), "Thu"),
        (dt.date(2022, 8, 5), "Fri"),
        (dt.date(2022, 8, 6), "Sat"),
        (dt.date(2022, 8, 7), "Sun"),
    ],
)
def test_weekday(d: dt.date, expected: str):
    actual = pk.weekday(d)
    assert actual == expected
