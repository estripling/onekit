import functools
import inspect
import itertools
import math
import os
import random
import re
import shutil
import string
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

import toolz
from toolz import curried

from onekit import timekit as tk

__all__ = (
    "BaseEnum",
    "ShellType",
    "archive_files",
    "are_predicates_true",
    "check_random_state",
    "coinflip",
    "concat_strings",
    "contrast_sets",
    "extend_range",
    "flatten",
    "filter_regex",
    "func_name",
    "get_shell_type",
    "headline",
    "highlight_string_differences",
    "lazy_read_lines",
    "map_regex",
    "num_to_str",
    "op",
    "parent_varname",
    "prompt_yes_no",
    "reduce_sets",
    "remove_punctuation",
    "signif",
    "source_code",
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
    name = name or f"{tk.timestamp(zone=timezone, fmt='%Y%m%d%H%M%S')}_{target.stem}"
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


def concat_strings(sep: str, /, *strings: str | None | Iterable[str | None]) -> str:
    """Concatenate strings, excluding None values.

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
    return sep.join(
        toolz.pipe(
            strings,
            flatten,
            curried.filter(lambda x: x is not None),
            curried.map(str),
        )
    )


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


def parent_varname(x: Any, /) -> str:
    """Returns the name of the parent variable of :math:`x`.

    Examples
    --------
    >>> from onekit import pythonkit as pk
    >>> my_var = "my_string_value"
    >>> def f(x) -> str:
    ...     return pk.parent_varname(x)
    ...
    >>> f(my_var)
    'my_var'
    """
    variables = inspect.currentframe().f_back.f_back.f_locals.items()
    return [name for name, value in variables if value is x][0]


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
