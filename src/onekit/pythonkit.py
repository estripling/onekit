import calendar
import datetime as dt
import functools
import inspect
import itertools
import math
import operator
import os
import random
import re
import shutil
import string
from contextlib import ContextDecorator
from enum import (
    Enum,
    member,
)
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    Iterator,
    Sequence,
)

import pytz
import toolz
from toolz import curried

__all__ = (
    "BaseEnum",
    "ShellType",
    "archive_files",
    "are_predicates_true",
    "check_random_state",
    "coinflip",
    "concat_strings",
    "contrast_sets",
    "date_ago",
    "date_ahead",
    "date_count_backward",
    "date_count_forward",
    "date_diff",
    "date_range",
    "date_to_str",
    "extend_range",
    "flatten",
    "filter_regex",
    "func_name",
    "get_shell_type",
    "headline",
    "highlight_string_differences",
    "humantime",
    "last_date_of_month",
    "lazy_read_lines",
    "map_regex",
    "num_days",
    "num_to_str",
    "op",
    "prompt_yes_no",
    "reduce_sets",
    "remove_punctuation",
    "signif",
    "source_code",
    "stopwatch",
    "str_to_date",
    "timestamp",
    "weekday",
)


Pair = tuple[float, float]
Predicate = Callable[[Any], bool]
Seed = int | random.Random | None


# noinspection PyTypeChecker
class BaseEnum(Enum):
    """Common methods for Enum classes."""

    @classmethod
    def all(cls) -> tuple[member]:
        """Returns tuple of all members of a concrete Enum class."""
        return tuple(mbr for mbr in cls)


class ShellType(BaseEnum):
    TERMINAL_INTERACTIVE_SHELL = "TerminalInteractiveShell"
    ZMQ_INTERACTIVE_SHELL = "ZMQInteractiveShell"


def archive_files(
    target: str,
    /,
    *,
    wildcards: list[str] | None = None,
    name: str | None = None,
    timezone: str | None = None,
    kind: str = "zip",
) -> None:
    """Archive files in target directory.

    Parameters
    ----------
    target : str
        Specify target directory to archive.
    wildcards : list of str, optional
        Specify wildcard to archive files.
        Default: all files in target directory are archived.
    name : str, optional
        Specify name of resulting archive.
        Default: name of target directory with timestamp.
    timezone : str, optional
        Specify timezone. Default: local timezone.
    kind : str, default="zip"
        Specify the archive type. Value is passed to the ``format`` argument of
        ``shutil.make_archive``, i.e., possible values are "zip", "tar",
        "gztar", "bztar", "xztar", or any other registered format.

    Returns
    -------
    NoneType
        Function has no return value. However, the archive of files of
        the target directory is stored in the current working directory.

    Examples
    --------
    >>> # archive all Python files and Notebooks in current working directory
    >>> from onekit import pythonkit as pk
    >>> pk.archive_files("./", wildcards=["*.py", "*.ipynb"])  # doctest: +SKIP
    """
    target = Path(target).resolve()
    wildcards = wildcards or ["**/*"]
    name = name or f"{timestamp(zone=timezone, fmt='%Y%m%d%H%M%S')}_{target.stem}"
    makedir = functools.partial(os.makedirs, exist_ok=True)

    with TemporaryDirectory() as tmpdir:
        for wildcard in wildcards:
            for src_file in target.rglob(wildcard):
                if os.path.isdir(src_file):
                    makedir(src_file)
                    continue

                dst_file = str(src_file).replace(str(target), tmpdir)
                dst_dir = str(src_file.parent).replace(str(target), tmpdir)
                makedir(dst_dir)
                shutil.copy(str(src_file), dst_file)

        shutil.make_archive(name, kind, tmpdir)


def are_predicates_true(
    func: Callable[..., bool],
    /,
    *predicates: Predicate | Iterable[Predicate],
) -> Predicate:
    """Evaluate if predicates are true.

    A predicate is of the form :math:`P\\colon X \\rightarrow \\{False, True\\}`

    Examples
    --------
    >>> from onekit import mathkit as mk
    >>> from onekit import pythonkit as pk
    >>> pk.are_predicates_true(all, lambda x: x % 2 == 0, lambda x: x % 5 == 0)(10)
    True

    >>> pk.are_predicates_true(all, lambda x: x % 2 == 0, lambda x: x % 5 == 0)(12)
    False

    >>> pk.are_predicates_true(any, lambda x: x % 2 == 0, lambda x: x % 5 == 0)(12)
    True

    >>> pk.are_predicates_true(any, lambda x: x % 2 == 0, lambda x: x % 5 == 0)(13)
    False

    >>> is_x_divisible_by_3_and_5 = pk.are_predicates_true(
    ...     all,
    ...     lambda x: mk.isdivisible(x, n=3),
    ...     lambda x: mk.isdivisible(x, n=5),
    ... )
    >>> type(is_x_divisible_by_3_and_5)
    <class 'function'>
    >>> is_x_divisible_by_3_and_5(60)
    True
    >>> is_x_divisible_by_3_and_5(9)
    False

    >>> is_x_divisible_by_3_or_5 = pk.are_predicates_true(
    ...     any,
    ...     lambda x: mk.isdivisible(x, n=3),
    ...     lambda x: mk.isdivisible(x, n=5),
    ... )
    >>> type(is_x_divisible_by_3_or_5)
    <class 'function'>
    >>> is_x_divisible_by_3_or_5(60)
    True
    >>> is_x_divisible_by_3_or_5(9)
    True
    >>> is_x_divisible_by_3_or_5(13)
    False
    """

    def inner(x: Any, /) -> bool:
        """Evaluate all specified predicates :math:`P_i` for value :math:`x \\in X`."""
        return func(predicate(x) for predicate in flatten(predicates))

    return inner


def check_random_state(seed: Seed = None) -> random.Random:
    """Turn seed into random.Random instance.

    Examples
    --------
    >>> import random
    >>> from onekit import pythonkit as pk
    >>> rng = pk.check_random_state()
    >>> isinstance(rng, random.Random)
    True
    """
    singleton_instance = getattr(random, "_inst")

    if seed is None or seed is singleton_instance:
        return singleton_instance

    elif isinstance(seed, int):
        return random.Random(seed)

    elif isinstance(seed, random.Random):
        return seed

    else:
        raise ValueError(f"{seed=} - cannot be used to seed Random instance")


def coinflip(bias: float, /, *, seed: Seed = None) -> bool:
    """Flip coin with adjustable bias.

    Examples
    --------
    >>> from functools import partial
    >>> from onekit import pythonkit as pk
    >>> {pk.coinflip(0.5) for _ in range(30)} == {True, False}
    True

    >>> fair_coin = partial(pk.coinflip, 0.5)
    >>> type(fair_coin)
    <class 'functools.partial'>
    >>> # fix coinflip outcome
    >>> fair_coin(seed=1)  # doctest: +SKIP
    True
    >>> # fix sequence of coinflip outcomes
    >>> rng = pk.check_random_state(2)
    >>> [fair_coin(seed=rng) for _ in range(6)]  # doctest: +SKIP
    [False, False, True, True, False, False]

    >>> biased_coin = partial(pk.coinflip, 0.6, seed=pk.check_random_state(3))
    >>> type(biased_coin)
    <class 'functools.partial'>
    >>> [biased_coin() for _ in range(6)]  # doctest: +SKIP
    [True, True, True, False, False, True]
    """
    if not (0 <= bias <= 1):
        raise ValueError(f"{bias=} - must be a float in [0, 1]")

    rng = check_random_state(seed)

    return rng.random() < bias


def concat_strings(sep: str, /, *strings: str | Iterable[str]) -> str:
    """Concatenate strings.

    Examples
    --------
    >>> from functools import partial
    >>> from onekit import pythonkit as pk
    >>> pk.concat_strings(" ", "Hello", "World")
    'Hello World'
    >>> pk.concat_strings(" ", ["Hello", "World"])
    'Hello World'

    >>> plus_concat = partial(pk.concat_strings, " + ")
    >>> plus_concat("Hello", "World")
    'Hello + World'
    >>> plus_concat(["Hello", "World"])
    'Hello + World'

    >>> # map onto list of lists of strings
    >>> ws_concat = partial(pk.concat_strings, " ")
    >>> list(map(ws_concat, [["Hello", "World"], ["Hi", "there"]]))
    ['Hello World', 'Hi there']
    """
    return sep.join(toolz.pipe(strings, flatten, curried.map(str)))


def contrast_sets(x: set, y: set, /, *, n: int = 3) -> dict:
    """Contrast sets.

    Examples
    --------
    >>> from onekit import pythonkit as pk
    >>> a = {"a", "c", "b", "g", "h", "i", "j", "k"}
    >>> b = {"c", "d", "e", "f", "g", "p", "q"}
    >>> summary = pk.contrast_sets(a, b)
    >>> isinstance(summary, dict)
    True
    >>> summary["x"] == a
    True
    >>> summary["y"] == b
    True
    >>> summary["x | y"] == a.union(b)
    True
    >>> summary["x & y"] == a.intersection(b)
    True
    >>> summary["x - y"] == a.difference(b)
    True
    >>> summary["y - x"] == b.difference(a)
    True
    >>> summary["x ^ y"] == a.symmetric_difference(b)
    True
    >>> print(summary["report"])
        x (n= 8): {'a', 'b', 'c', ...}
        y (n= 7): {'c', 'd', 'e', ...}
    x | y (n=13): {'a', 'b', 'c', ...}
    x & y (n= 2): {'c', 'g'}
    x - y (n= 6): {'a', 'b', 'h', ...}
    y - x (n= 5): {'d', 'e', 'f', ...}
    x ^ y (n=11): {'a', 'b', 'd', ...}
    jaccard = 0.153846
    overlap = 0.285714
    dice = 0.266667
    disjoint?: False
    x == y: False
    x <= y: False
    x <  y: False
    y <= x: False
    y <  x: False
    """
    x, y = set(x), set(y)
    union = x.union(y)
    intersection = x.intersection(y)
    in_x_but_not_y = x.difference(y)
    in_y_but_not_x = y.difference(x)
    symmetric_diff = x ^ y
    jaccard = len(intersection) / len(union)
    overlap = len(intersection) / min(len(x), len(y))
    dice = 2 * len(intersection) / (len(x) + len(y))

    output = {
        "x": x,
        "y": y,
        "x | y": union,
        "x & y": intersection,
        "x - y": in_x_but_not_y,
        "y - x": in_y_but_not_x,
        "x ^ y": symmetric_diff,
        "jaccard": jaccard,
        "overlap": overlap,
        "dice": dice,
    }

    max_set_size = max(
        len(num_to_str(len(v))) for v in output.values() if isinstance(v, set)
    )

    lines = []
    for k, v in output.items():
        if isinstance(v, set):
            elements = f"{sorted(v)[:n]}".replace("[", "{")
            elements = (
                elements.replace("]", ", ...}")
                if len(v) > n
                else elements.replace("]", "}")
            )
            elements = elements.replace(",", "") if len(v) == 1 else elements

            set_size = num_to_str(len(v)).rjust(max_set_size)
            desc = f"{k} (n={set_size})"

            if k in ["x", "y"]:
                desc = f"    {desc}"
            msg = f"{desc}: {elements}"
            lines.append(msg)

        else:
            lines.append(f"{k} = {v:g}")

    tmp = {
        "disjoint?": x.isdisjoint(y),
        "x == y": x == y,
        "x <= y": x <= y,
        "x <  y": x < y,
        "y <= x": y <= x,
        "y <  x": y < x,
    }

    for k, v in tmp.items():
        lines.append(f"{k}: {v}")

    output.update(tmp)
    output["report"] = "\n".join(lines)

    return output


def date_ago(ref_date: dt.date, /, n: int) -> dt.date:
    """Compute date :math:`n \\in \\mathbb{N}_{0}` days ago from reference date.

    Examples
    --------
    >>> import datetime as dt
    >>> from onekit import pythonkit as pk
    >>> ref_date = dt.date(2022, 1, 1)
    >>> pk.date_ago(ref_date, n=0)
    datetime.date(2022, 1, 1)
    >>> pk.date_ago(ref_date, n=1)
    datetime.date(2021, 12, 31)
    >>> pk.date_ago(ref_date, n=2)
    datetime.date(2021, 12, 30)
    """
    if not isinstance(n, int) or n < 0:
        raise ValueError(f"{n=} - must be a non-negative integer")
    return ref_date - dt.timedelta(days=n)


def date_ahead(ref_date: dt.date, /, n: int) -> dt.date:
    """Compute date :math:`n \\in \\mathbb{N}_{0}` days ahead from reference date.

    Examples
    --------
    >>> import datetime as dt
    >>> from onekit import pythonkit as pk
    >>> ref_date = dt.date(2022, 1, 1)
    >>> pk.date_ahead(ref_date, n=0)
    datetime.date(2022, 1, 1)
    >>> pk.date_ahead(ref_date, n=1)
    datetime.date(2022, 1, 2)
    >>> pk.date_ahead(ref_date, n=2)
    datetime.date(2022, 1, 3)
    """
    if not isinstance(n, int) or n < 0:
        raise ValueError(f"{n=} - must be a non-negative integer")
    return ref_date + dt.timedelta(days=n)


def date_count_backward(ref_date: dt.date, /) -> Generator:
    """Generate sequence of consecutive dates in backward manner w.r.t. reference date.

    Examples
    --------
    >>> import datetime as dt
    >>> from toolz import curried
    >>> from onekit import pythonkit as pk
    >>> ref_date = dt.date(2022, 1, 1)
    >>> curried.pipe(
    ...     pk.date_count_backward(ref_date),
    ...     curried.map(pk.date_to_str),
    ...     curried.take(3),
    ...     list,
    ... )
    ['2022-01-01', '2021-12-31', '2021-12-30']
    """
    successor = operator.sub
    return toolz.iterate(lambda d: successor(d, dt.timedelta(1)), ref_date)


def date_count_forward(ref_date: dt.date, /) -> Generator:
    """Generate sequence of consecutive dates in forward manner w.r.t. reference date.

    Examples
    --------
    >>> import datetime as dt
    >>> from toolz import curried
    >>> from onekit import pythonkit as pk
    >>> ref_date = dt.date(2022, 1, 1)
    >>> curried.pipe(
    ...     pk.date_count_forward(ref_date),
    ...     curried.map(pk.date_to_str),
    ...     curried.take(3),
    ...     list,
    ... )
    ['2022-01-01', '2022-01-02', '2022-01-03']
    """
    successor = operator.add
    return toolz.iterate(lambda d: successor(d, dt.timedelta(1)), ref_date)


def date_diff(from_date: dt.date, to_date: dt.date, /) -> int:
    """Compute difference betweem dates.

    Examples
    --------
    >>> import datetime as dt
    >>> from onekit import pythonkit as pk
    >>> d1 = dt.date(2024, 7, 1)
    >>> d2 = dt.date(2024, 7, 7)

    >>> pk.date_diff(d1, d1)
    0

    >>> pk.date_diff(d1, d2)
    6

    >>> pk.date_diff(d2, d1)
    -6
    """
    return (to_date - from_date).days


# noinspection PyTypeChecker
def date_range(
    min_date: dt.date,
    max_date: dt.date,
    /,
    *,
    incl_min: bool = True,
    incl_max: bool = True,
) -> Generator:
    """Generate sequence of consecutive dates between two dates.

    Examples
    --------
    >>> import datetime as dt
    >>> from toolz import curried
    >>> from onekit import pythonkit as pk
    >>> d1 = dt.date(2022, 1, 1)
    >>> d2 = dt.date(2022, 1, 3)

    >>> curried.pipe(pk.date_range(d1, d2), curried.map(pk.date_to_str), list)
    ['2022-01-01', '2022-01-02', '2022-01-03']

    >>> curried.pipe(
    ...     pk.date_range(d1, d2, incl_min=False, incl_max=True),
    ...     curried.map(pk.date_to_str),
    ...     list,
    ... )
    ['2022-01-02', '2022-01-03']

    >>> curried.pipe(
    ...     pk.date_range(d1, d2, incl_min=True, incl_max=False),
    ...     curried.map(pk.date_to_str),
    ...     list,
    ... )
    ['2022-01-01', '2022-01-02']

    >>> curried.pipe(
    ...     pk.date_range(d1, d2, incl_min=False, incl_max=False),
    ...     curried.map(pk.date_to_str),
    ...     list,
    ... )
    ['2022-01-02']

    >>> list(pk.date_range(d1, dt.date(2022, 1, 1)))
    [datetime.date(2022, 1, 1)]

    >>> list(pk.date_range(d1, dt.date(2022, 1, 1), incl_min=False))
    []

    >>> # function makes sure: start <= end
    >>> curried.pipe(pk.date_range(d2, d1), curried.map(pk.date_to_str), list)
    ['2022-01-01', '2022-01-02', '2022-01-03']
    """
    d1, d2 = sorted([min_date, max_date])
    d1 = d1 if incl_min else d1 + dt.timedelta(1)
    d2 = d2 if incl_max else d2 - dt.timedelta(1)
    return itertools.takewhile(lambda d: d <= d2, date_count_forward(d1))


def date_to_str(d: dt.date, /) -> str:
    """Cast date to string in ISO format: YYYY-MM-DD.

    Examples
    --------
    >>> import datetime as dt
    >>> from onekit import pythonkit as pk
    >>> pk.date_to_str(dt.date(2022, 1, 1))
    '2022-01-01'
    """
    return d.isoformat()


def extend_range(xmin: float, xmax: float, /, *, factor: float = 0.05) -> Pair:
    """Extend value range ``xmax - xmin`` by factor.

    Examples
    --------
    >>> from onekit import pythonkit as pk
    >>> pk.extend_range(0.0, 1.0)
    (-0.05, 1.05)

    >>> pk.extend_range(0.0, 1.0, factor=0.1)
    (-0.1, 1.1)
    """
    if not isinstance(factor, float) or factor < 0:
        raise ValueError(f"{factor=} - must be a non-negative float")

    xmin, xmax = sorted([xmin, xmax])
    value_range = xmax - xmin

    new_xmin = xmin - factor * value_range
    new_xmax = xmax + factor * value_range

    return new_xmin, new_xmax


# noinspection PyTypeChecker
def filter_regex(
    pattern: str,
    /,
    *strings: str | Iterable[str],
    flags=re.IGNORECASE,
) -> Generator:
    """Filter iterable of strings with regex.

    Examples
    --------
    >>> from functools import partial
    >>> from onekit import pythonkit as pk
    >>> list(pk.filter_regex("hello", "Hello, World!", "Hi, there!", "Hello!"))
    ['Hello, World!', 'Hello!']

    >>> strings = [
    ...     "Guiding principles for Python's design: The Zen of Python",
    ...     "Beautiful is better than ugly.",
    ...     "Explicit is better than implicit.",
    ...     "Simple is better than complex.",
    ... ]
    >>> list(pk.filter_regex("python", strings))
    ["Guiding principles for Python's design: The Zen of Python"]

    >>> filter_regex__hi = partial(pk.filter_regex, "hi")
    >>> list(filter_regex__hi("Hello, World!", "Hi, there!", "Hello!"))
    ['Hi, there!']
    """
    return filter(functools.partial(re.findall, pattern, flags=flags), flatten(strings))


def flatten(*items: Any | Iterable[Any]) -> Generator:
    """Flatten iterable of items.

    Examples
    --------
    >>> from onekit import pythonkit as pk
    >>> list(pk.flatten([[1, 2], *[3, 4], [5]]))
    [1, 2, 3, 4, 5]

    >>> list(pk.flatten([1, (2, 3)], 4, [], [[[5]], 6]))
    [1, 2, 3, 4, 5, 6]

    >>> list(pk.flatten(["one", 2], 3, [(4, "five")], [[["six"]]], "seven", []))
    ['one', 2, 3, 4, 'five', 'six', 'seven']
    """

    def _flatten(items):
        for item in items:
            if isinstance(item, (Iterator, Sequence)) and not isinstance(item, str):
                yield from _flatten(item)
            else:
                yield item

    return _flatten(items)


def func_name() -> str:
    """Get name of called function.

    Examples
    --------
    >>> from onekit import pythonkit as pk
    >>> def foobar():
    ...     return pk.func_name()
    ...
    >>> foobar()
    'foobar'
    """
    return inspect.stack()[1].function


def get_shell_type() -> str:  # pragma: no cover
    """Returns the type of the current shell session.

    The function identifies whether code is executed from within
    a 'python', 'ipython', or 'notebook' session.

    Examples
    --------
    >>> from onekit import pythonkit as pk
    >>> pk.get_shell_type()
    'python'
    """
    try:
        from IPython import get_ipython

        if get_ipython() is None:
            return "python"

        shell = get_ipython().__class__.__name__

        if shell == ShellType.TERMINAL_INTERACTIVE_SHELL.value:
            return "ipython"

        elif shell == ShellType.ZMQ_INTERACTIVE_SHELL.value:
            return "notebook"

        else:
            return "python"

    except (ModuleNotFoundError, ImportError, NameError):
        return "python"


def headline(text: str, /, *, n: int = 88, fillchar: str = "-") -> str:
    """Create headline string.

    Examples
    --------
    >>> from onekit import pythonkit as pk
    >>> pk.headline("Hello, World!", n=30)
    '------- Hello, World! --------'
    """
    return f" {text} ".center(n, fillchar)


def highlight_string_differences(lft_str: str, rgt_str: str, /) -> str:
    """Highlight differences between two strings.

    Examples
    --------
    >>> from onekit import pythonkit as pk
    >>> print(pk.highlight_string_differences("hello", "hall"))  # doctest: +SKIP
    hello
     |  |
    hall

    >>> # no differences when there is no '|' character
    >>> print(pk.highlight_string_differences("hello", "hello"))  # doctest: +SKIP
    hello
    <BLANKLINE>
    hello
    """
    return concat_strings(
        os.linesep,
        lft_str,
        concat_strings(
            "",
            *(
                " " if x == y else "|"
                for x, y in itertools.zip_longest(lft_str, rgt_str, fillvalue="")
            ),
        ),
        rgt_str,
    )


def humantime(seconds: int | float, /) -> str:
    """Convert seconds to human-readable time.

    Examples
    --------
    >>> from onekit import pythonkit as pk
    >>> # 1 second
    >>> pk.humantime(1)
    '1s'

    >>> # 1 minute
    >>> pk.humantime(60)
    '1m'

    >>> # 1 hour
    >>> pk.humantime(60 * 60)
    '1h'

    >>> # 1 day
    >>> pk.humantime(60 * 60 * 24)
    '1d'

    >>> pk.humantime(60 * 60 * 24 + 60 * 60 + 60 + 1)
    '1d 1h 1m 1s'

    >>> pk.humantime(3 * 60 * 60 * 24 + 2 * 60)
    '3d 2m'
    """
    if seconds < 0:
        raise ValueError(f"{seconds=} - must be a non-negative number")

    if math.isclose(seconds, 0):
        return "0s"

    if 0 < seconds < 60:
        return f"{seconds:g}s"

    minutes, seconds = divmod(int(round(seconds, 0)), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)

    output = []
    if days:
        output.append(f"{days}d")

    if hours:
        output.append(f"{hours}h")

    if minutes:
        output.append(f"{minutes}m")

    if seconds:
        output.append(f"{seconds}s")

    return " ".join(output)


def last_date_of_month(year: int, month: int, /) -> dt.date:
    """Get the last date of the month.

    Examples
    --------
    >>> from onekit import pythonkit as pk
    >>> pk.last_date_of_month(2022, 1)
    datetime.date(2022, 1, 31)
    """
    _, number_of_days_in_month = calendar.monthrange(year, month)
    return dt.date(year, month, number_of_days_in_month)


def lazy_read_lines(
    path: str | Path,
    /,
    *,
    encoding: str | None = None,
    errors: str | None = None,
    newline: str | None = None,
) -> Generator:
    """Lazily read text file line by line.

    Examples
    --------
    >>> import inspect
    >>> from toolz import curried
    >>> from onekit import pythonkit as pk
    >>> inspect.isgeneratorfunction(pk.lazy_read_lines)
    True

    >>> text_lines = curried.pipe(  # doctest: +SKIP
    ...     pk.lazy_read_lines("./my_text_file.txt"),
    ...     curried.map(str.rstrip),
    ... )
    """
    with open(
        file=str(path),
        mode="r",
        encoding=encoding,
        errors=errors,
        newline=newline,
    ) as lines:
        for line in lines:
            yield line


# noinspection PyTypeChecker
def map_regex(
    pattern: str,
    /,
    *strings: str | Iterable[str],
    flags=re.IGNORECASE,
) -> Generator:
    """Match regex to iterable of strings.

    Examples
    --------
    >>> from functools import partial
    >>> from onekit import pythonkit as pk
    >>> list(pk.map_regex("hello", "Hello, World!", "Hi, there!", "Hello!"))
    [['Hello'], [], ['Hello']]

    >>> strings = [
    ...     "Guiding principles for Python's design: The Zen of Python",
    ...     "Beautiful is better than ugly.",
    ...     "Explicit is better than implicit.",
    ...     "Simple is better than complex.",
    ... ]
    >>> list(pk.map_regex("python", strings))
    [['Python', 'Python'], [], [], []]

    >>> map_regex__hi = partial(pk.map_regex, "hi")
    >>> list(map_regex__hi("Hello, World!", "Hi, there!", "Hello!"))
    [[], ['Hi'], []]
    """
    return map(functools.partial(re.findall, pattern, flags=flags), flatten(strings))


def num_days(d1: dt.date, d2: dt.date, /) -> int:
    """Compute the number of days between two dates (both endpoints inclusive).

    Examples
    --------
    >>> import datetime as dt
    >>> from onekit import pythonkit as pk
    >>> pk.num_days(dt.date(2022, 8, 1), dt.date(2022, 8, 1))
    1

    >>> pk.num_days(dt.date(2022, 8, 1), dt.date(2022, 8, 2))
    2

    >>> pk.num_days(dt.date(2022, 8, 1), dt.date(2022, 8, 7))
    7

    >>> # function makes sure: start <= end
    >>> pk.num_days(dt.date(2022, 8, 7), dt.date(2022, 8, 1))
    7
    """
    return abs(date_diff(d1, d2)) + 1


def num_to_str(x: int | float, /) -> str:
    """Cast number to string with underscores as thousands separator.

    Examples
    --------
    >>> from onekit import pythonkit as pk
    >>> pk.num_to_str(1000000)
    '1_000_000'

    >>> pk.num_to_str(100000.0)
    '100_000'
    """
    f, i = math.modf(x)
    integer = f"{int(i):_}"
    fractional = f"{f:g}".lstrip("0")
    return integer if math.isclose(f, 0) else f"{integer}{fractional}"


def op(func: Callable, const: Any, /) -> Callable[[Any], Any]:
    """Leverage operator functions.

    Use ``op`` to create functions of ``x`` with a fixed argument ``const``.

    Examples
    --------
    >>> import operator
    >>> from onekit import pythonkit as pk
    >>> inc = pk.op(operator.add, 1)
    >>> inc(1)
    2

    >>> dec = pk.op(operator.sub, 1)
    >>> dec(1)
    0
    """

    def inner(x: Any, /) -> Any:
        return func(x, const)

    return inner


def prompt_yes_no(question: str, /, *, default: str | None = None) -> bool:
    """Prompt yes-no question.

    Examples
    --------
    >>> from onekit import pythonkit as pk
    >>> pk.prompt_yes_no("Is all clear?")  # doctest: +SKIP
    Is all clear? [y/n] y<enter>
    True

    >>> pk.prompt_yes_no("Do you like onekit?", default="yes")  # doctest: +SKIP
    Do you like onekit? [Y/n] <enter>
    True

    >>> pk.prompt_yes_no("Do you like onekit?", default="yes")  # doctest: +SKIP
    Do you like onekit? [Y/n] yay<enter>
    Do you like onekit? Please respond with 'yes' [Y] or 'no' [n] <enter>
    True
    """
    prompt = (
        "[y/n]"
        if default is None
        else "[Y/n]" if default == "yes" else "[y/N]" if default == "no" else "invalid"
    )

    if prompt == "invalid":
        raise ValueError(f"{default=} - must be either None, 'yes', or 'no'")

    answer = input(f"{question} {prompt} ").lower()

    def strtobool(value: str) -> bool:
        """Convert a string representation of truth to true (1) or false (0).

        True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
        are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
        'val' is anything else.

        Notes
        -----
        - Shamelessly copied and modified from: distutils.util.strtobool
        - distutils is not available with Python>=3.12
        """
        value = value.lower()
        if value in ("y", "yes", "t", "true", "on", "1"):
            return True
        elif value in ("n", "no", "f", "false", "off", "0"):
            return False
        else:
            raise ValueError("invalid truth value {!r}".format(value))

    while True:
        try:
            if answer == "" and default in ["yes", "no"]:
                return bool(strtobool(default))
            return bool(strtobool(answer))

        except ValueError:
            response_text = "{} Please respond with 'yes' [{}] or 'no' [{}] ".format(
                question,
                "Y" if default == "yes" else "y",
                "N" if default == "no" else "n",
            )
            answer = input(response_text).lower()


# noinspection PyTypeChecker
def reduce_sets(func: Callable[[set, set], set], /, *sets: set | Iterable[set]) -> set:
    """Apply function of two set arguments to reduce iterable of sets.

    Examples
    --------
    >>> from onekit import pythonkit as pk
    >>> x = {0, 1, 2, 3}
    >>> y = {2, 4, 6}
    >>> z = {2, 6, 8}
    >>> pk.reduce_sets(set.intersection, x, y, z)
    {2}
    >>> sets = [x, y, z]
    >>> pk.reduce_sets(set.symmetric_difference, sets)
    {0, 1, 2, 3, 4, 8}
    >>> pk.reduce_sets(set.difference, *sets)
    {0, 1, 3}

    >>> pk.reduce_sets(set.union, x, y, z)
    {0, 1, 2, 3, 4, 6, 8}
    >>> pk.reduce_sets(set.union, sets)
    {0, 1, 2, 3, 4, 6, 8}
    >>> pk.reduce_sets(set.union, *sets)
    {0, 1, 2, 3, 4, 6, 8}
    """
    return toolz.pipe(sets, flatten, curried.map(set), curried.reduce(func))


def remove_punctuation(text: str, /) -> str:
    """Remove punctuation from text string.

    Examples
    --------
    >>> from onekit import pythonkit as pk
    >>> pk.remove_punctuation("I think, therefore I am. --Descartes")
    'I think therefore I am Descartes'
    """
    return text.translate(str.maketrans("", "", string.punctuation))


def signif(x: int | float, /, n: int) -> int | float:
    """Round :math:`x` to its :math:`n` significant digits.

    Examples
    --------
    >>> from onekit import pythonkit as pk
    >>> pk.signif(987654321, 3)
    988000000

    >>> [pk.signif(14393237.76, n) for n in range(1, 6)]
    [10000000.0, 14000000.0, 14400000.0, 14390000.0, 14393000.0]

    >>> pk.signif(14393237.76, n=3)
    14400000.0
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"{n=} - must be a positive integer")

    if not math.isfinite(x) or math.isclose(x, 0.0):
        return x

    n -= math.ceil(math.log10(abs(x)))
    return round(x, n)


def source_code(x: Any, /) -> str:
    """Get source code of an object :math:`x`.

    Examples
    --------
    >>> from onekit import pythonkit as pk
    >>> def greet():
    ...     return "Hello, World!"
    ...
    >>> print(pk.source_code(greet))
    def greet():
        return "Hello, World!"
    <BLANKLINE>
    """
    return inspect.getsource(x)


class stopwatch(ContextDecorator):
    """Measure elapsed wall-clock time and print it to standard output.

    Parameters
    ----------
    label : str, int, optional
        Specify label. When used as a decorator and label is not specified,
        label is the name of the function.
    flush : bool, default=False
        Passed to built-in print function:
         - If ``True``, prints start time before stop time.
         - If ``False``, prints start time and stop time all at once.
    timezone : str, optional
        Specify timezone. Default: local timezone.
    fmt : str, optional
        Specify timestamp format. Default: ``%Y-%m-%d %H:%M:%S``.

    Notes
    -----
    - Instantiation and use of an instance's properties is only possible
      when ``stopwatch`` is used as a context manager (see examples).
    - The total elapsed time is computed when multiple ``stopwatch`` instances
      are added (see examples).

    Examples
    --------
    >>> # as context manager
    >>> import time
    >>> from onekit import pythonkit as pk
    >>> with pk.stopwatch():  # doctest: +SKIP
    ...     time.sleep(0.1)
    ...
    2023-01-01 12:00:00 -> 2023-01-01 12:00:00 = 0.100691s

    >>> # as decorator
    >>> import time
    >>> from onekit import pythonkit as pk
    >>> @pk.stopwatch()
    ... def func():
    ...     time.sleep(0.1)
    ...
    >>> func()  # doctest: +SKIP
    2023-01-01 12:00:00 -> 2023-01-01 12:00:00 = 0.100709s - func

    >>> # stopwatch instance
    >>> import time
    >>> from onekit import pythonkit as pk
    >>> with pk.stopwatch("instance-example") as sw:  # doctest: +SKIP
    ...     time.sleep(0.1)
    ...
    2023-01-01 12:00:00 -> 2023-01-01 12:00:00 = 0.100647s - instance-example
    >>> sw.label  # doctest: +SKIP
    'instance-example'
    >>> sw.flush  # doctest: +SKIP
    True
    >>> sw.fmt  # doctest: +SKIP
    '%Y-%m-%d %H:%M:%S'
    >>> sw.start_time  # doctest: +SKIP
    datetime.datetime(2023, 1, 1, 12, 0, 0, 732176)
    >>> sw.stop_time  # doctest: +SKIP
    datetime.datetime(2023, 1, 1, 12, 0, 0, 832823)
    >>> sw.elapsed_time  # doctest: +SKIP
    datetime.timedelta(microseconds=100647)
    >>> sw  # doctest: +SKIP
    2023-01-01 12:00:00 -> 2023-01-01 12:00:00 = 0.100647s - instance-example

    >>> # compute total elapsed time
    >>> import time
    >>> from onekit import pythonkit as pk
    >>> with pk.stopwatch(1) as sw1:  # doctest: +SKIP
    ...     time.sleep(1)
    ...
    2023-01-01 12:00:00 -> 2023-01-01 12:00:01 = 1.00122s - 1
    >>> with pk.stopwatch(2) as sw2:  # doctest: +SKIP
    ...     time.sleep(1)
    ...
    2023-01-01 12:01:00 -> 2023-01-01 12:01:01 = 1.00121s - 2
    >>> with pk.stopwatch(3) as sw3:  # doctest: +SKIP
    ...     time.sleep(1)
    ...
    2023-01-01 12:02:00 -> 2023-01-01 12:02:01 = 1.00119s - 3
    >>> sw1 + sw2 + sw3  # doctest: +SKIP
    3.00362s - total elapsed time
    >>> sum([sw1, sw2, sw3])  # doctest: +SKIP
    3.00362s - total elapsed time
    """

    def __init__(
        self,
        label: str | int | None = None,
        /,
        *,
        flush: bool = False,
        timezone: str | None = None,
        fmt: str | None = None,
    ):
        if isinstance(label, bool) or (
            label is not None and not isinstance(label, (str, int))
        ):
            raise TypeError(f"{label=} - must be str, int, or NoneType")

        if not isinstance(flush, bool):
            raise TypeError(f"{flush=} - must be bool")

        if timezone is not None and not isinstance(timezone, str):
            raise TypeError(f"{timezone=} - must be str or NoneType")

        if fmt is not None and not isinstance(fmt, str):
            raise TypeError(f"{fmt=} - must be str or NoneType")

        self._label = label
        self._flush = flush
        self._timezone = timezone
        self._fmt = "%Y-%m-%d %H:%M:%S" if fmt is None else fmt
        self._start_time = None
        self._stop_time = None
        self._elapsed_time = None
        self._is_total = False

    def __repr__(self):
        return (
            super().__repr__() if self.elapsed_time is None else self._output_message()
        )

    @property
    def label(self):
        """Retrieve label value.

        Returns
        -------
        NoneType or str
            Label if specified in the call else None when used as context manager.
            When used as decorator, label is the name of the decorated function.
        """
        return self._label

    @property
    def flush(self):
        """Retrieve flush value.

        Returns
        -------
        bool
            Value used in the built-in function when printing to standard output.
        """
        return self._flush

    @property
    def timezone(self):
        """Retrieve timezone value.

        Returns
        -------
        str
            Value used for timezone.
        """
        return self._timezone

    @property
    def fmt(self):
        """Retrieve timestamp format.

        The timestamp format can be changed by passing a new value that is accepted
        by ``strftime``. Note that the underlying data remain unchanged.

        Returns
        -------
        str
            Format to use to convert a ``datetime`` object to a string via ``strftime``.
        """
        return self._fmt

    @fmt.setter
    def fmt(self, value):
        if not isinstance(value, str):
            raise TypeError(f"{value=} - `fmt` must be str")
        self._fmt = value

    @property
    def start_time(self):
        """Retrieve start time value.

        Returns
        -------
        datetime.datetime
            Timestamp of the start time.
        """
        return self._start_time

    @property
    def stop_time(self):
        """Retrieve stop time value.

        Returns
        -------
        datetime.datetime
            Timestamp of the stop time.
        """
        return self._stop_time

    @property
    def elapsed_time(self):
        """Retrieve elapsed time value.

        Returns
        -------
        datetime.timedelta
            The elapsed time between start and stop.
        """
        return self._elapsed_time

    def __call__(self, func):
        if self.label is None:
            self._label = func.__name__
        return super().__call__(func)

    def __enter__(self):
        self._start_time = dt.datetime.now(
            tz=None if self.timezone is None else pytz.timezone(self.timezone)
        )
        if self.flush:
            print(self._message_part_1(), end="", flush=True)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._stop_time = dt.datetime.now(
            tz=None if self.timezone is None else pytz.timezone(self.timezone)
        )
        self._elapsed_time = self.stop_time - self.start_time
        print(self._message_part_2() if self.flush else self._output_message())
        return False

    def _output_message(self):
        return (
            f"{self._human_readable_elapsed_time()} - {self.label}"
            if self._is_total
            else self._message_part_1() + self._message_part_2()
        )

    def _message_part_1(self):
        return self._datetime_to_str(self.start_time) + " -> "

    def _message_part_2(self):
        suffix = "" if self.label is None else f" - {self.label}"
        return (
            self._datetime_to_str(self.stop_time)
            + " = "
            + self._human_readable_elapsed_time()
            + suffix
        )

    def _human_readable_elapsed_time(self):
        if self.elapsed_time is not None:
            return humantime(self.elapsed_time.total_seconds())

    def _datetime_to_str(self, date_time):
        return date_time.strftime(self.fmt)

    def __add__(self, other):
        total = self._create_total_instance()
        total._elapsed_time = self.elapsed_time + other.elapsed_time
        return total

    def __radd__(self, other):
        other_elapsed_time = (
            other.elapsed_time if isinstance(other, stopwatch) else dt.timedelta()
        )
        total = self._create_total_instance()
        total._elapsed_time = other_elapsed_time + self.elapsed_time
        return total

    @staticmethod
    def _create_total_instance():
        total = stopwatch("total elapsed time", fmt=None, flush=False)
        total._fmt = None
        total._is_total = True
        return total


def str_to_date(string: str, /) -> dt.date:
    """Cast ISO date string to date.

    Examples
    --------
    >>> from onekit import pythonkit as pk
    >>> pk.str_to_date("2022-01-01")
    datetime.date(2022, 1, 1)
    """
    return dt.datetime.strptime(string, "%Y-%m-%d").date()


def timestamp(zone: str | None = None, fmt: str | None = None) -> str:
    """Get timezone-dependent timestamp.

    Parameters
    ----------
    zone : str, optional
        Specify timezone. Default: local timezone.
    fmt : str, optional
        Specify timestamp format. Default: ``%Y-%m-%d %H:%M:%S``.

    Notes
    -----
    - Look up available timezones: ``pytz.all_timezones`` ``pytz.common_timezones``
    - Look up timezones per country:  ``pytz.country_names`` ``pytz.country_timezones``

    Examples
    --------
    >>> from onekit import pythonkit as pk
    >>> pk.timestamp()  # doctest: +SKIP
    '2024-01-01 00:00:00'

    >>> pk.timestamp("CET")  # doctest: +SKIP
    '2024-01-01 01:00:00'
    """
    zone = None if zone is None else pytz.timezone(zone)
    fmt = fmt or "%Y-%m-%d %H:%M:%S"
    return dt.datetime.now(tz=zone).strftime(fmt)


def weekday(d: dt.date, /) -> str:
    """Get name of the weekday.

    Examples
    --------
    >>> import datetime as dt
    >>> from onekit import pythonkit as pk
    >>> pk.weekday(dt.date(2022, 8, 1))
    'Mon'
    >>> pk.weekday(dt.date(2022, 8, 7))
    'Sun'
    """
    return d.strftime("%a")
