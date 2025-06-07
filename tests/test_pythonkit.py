import functools
import math
import operator
import os
import random
from enum import auto
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
    Any,
    Callable,
    Sequence,
)

import pytest
import toolz
from toolz import curried

from onekit import pythonkit as pk
from onekit.pythonkit import (
    BaseEnum,
    ShellType,
)


def test_base_enum():
    class Animal(BaseEnum):
        CAT = auto()
        DOG = auto()

    actual = Animal.all()
    expected = (Animal.CAT, Animal.DOG)
    assert actual == expected


def test_shell_type():
    assert ShellType.TERMINAL_INTERACTIVE_SHELL.value == "TerminalInteractiveShell"
    assert ShellType.ZMQ_INTERACTIVE_SHELL.value == "ZMQInteractiveShell"
    assert len(ShellType.all()) == 2


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

    strings = ["Hello", None, "World", None]
    actual = pk.concat_strings(" ", strings)
    expected = "Hello World"
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
    expected: tuple[float, float],
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
def test_flatten(items: Any, expected: list[Any]):
    actual = list(pk.flatten(items))
    assert actual == expected


def test_func_name():
    def foobar():
        return pk.func_name()

    actual = foobar()
    expected = "foobar"
    assert actual == expected


def test_get_shell_type():
    actual = pk.get_shell_type()
    expected = "python"
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
    actual = pk.highlight_string_differences(lft_str, rgt_str).replace(os.linesep, "\n")
    assert actual == expected


def test_lazy_read_lines():
    expected = ("one", "two", "three")

    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir).joinpath("test_file_for_lazy_read_lines.txt")

        with path.open("w") as fh:
            fh.write(pk.concat_strings(os.linesep, expected))

        actual = toolz.pipe(
            pk.lazy_read_lines(path),
            curried.map(str.rstrip),
            curried.filter(lambda x: len(x) > 0),
            tuple,
        )

        assert actual == expected

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
        (100000.0, "100_000"),
        (100000.01, "100_000.01"),
    ],
)
def test_num_to_str(x: int | float, expected: str):
    actual = pk.num_to_str(x)
    assert actual == expected


def test_parent_varname():
    my_var = None

    def f(x) -> str:
        return pk.parent_varname(x)

    actual = f(my_var)
    expected = "my_var"
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
def test_reduce_sets(func: Callable, expected: set[int]):
    x = {0, 1, 2, 3}
    y = {2, 4, 6}
    z = {2, 6, 8}

    f = functools.partial(pk.reduce_sets, func)

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
def test_signif(x: int | float, n: int, expected: int | float):
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
        default: str | None,
        answer: str,
        expected: bool,
    ):
        monkeypatch.setattr("sys.stdin", StringIO(answer))
        actual = pk.prompt_yes_no("Do you like onekit?", default=default)
        assert actual == expected

    @pytest.mark.parametrize("default", [1, "noo", "yeah"])
    def test_invalid_default_value(self, default: str | None):
        with pytest.raises(ValueError):
            pk.prompt_yes_no("Do you like onekit?", default=default)

    def test_subsequent_prompt(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", StringIO("yay"))
        with pytest.raises(EOFError):
            pk.prompt_yes_no("Do you like onekit?", default="yes")


class TestRegexFunctions:
    def test_filter_regex(self, zen_of_python: tuple[str]):
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

    def test_map_regex(self, zen_of_python: tuple[str]):
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
    def zen_of_python(self) -> Sequence[str]:
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
