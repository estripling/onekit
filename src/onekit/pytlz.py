"""Python toolz."""

import datetime as dt
import inspect
import math
import random
from typing import (
    Any,
    Callable,
    Generator,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import toolz
from toolz.curried import (
    map,
    reduce,
)

__all__ = (
    "all_predicate_true",
    "any_predicate_true",
    "check_random_state",
    "coinflip",
    "collatz",
    "contrast_sets",
    "date_to_str",
    "extend_range",
    "fibonacci",
    "flatten",
    "func_name",
    "isdivisibleby",
    "iseven",
    "isodd",
    "num_to_str",
    "reduce_sets",
    "signif",
    "source_code",
)


Pair = Tuple[float, float]
Predicate = Callable[[Any], bool]
Seed = Optional[Union[int, random.Random]]


def all_predicate_true(*predicates: Sequence[Predicate]) -> Predicate:
    """Evaluate if every predicate is true.

    A predicate is of the form :math:`P\\colon X \\rightarrow \\{False, True\\}`

    Examples
    --------
    >>> from onekit import pytlz
    >>> pytlz.all_predicate_true(lambda x: x % 2 == 0, lambda x: x % 5 == 0)(10)
    True

    >>> is_divisible_by_3_and_5 = pytlz.all_predicate_true(
    ...     pytlz.isdivisibleby(3),
    ...     pytlz.isdivisibleby(5),
    ... )
    >>> type(is_divisible_by_3_and_5)
    <class 'function'>
    >>> is_divisible_by_3_and_5(60)
    True
    >>> is_divisible_by_3_and_5(9)
    False
    """

    def inner(x: Any, /) -> bool:
        """Evaluate all specified predicates :math:`P_i` for value :math:`x \\in X`."""
        return all(predicate(x) for predicate in flatten(predicates))

    return inner


def any_predicate_true(*predicates: Sequence[Predicate]) -> Predicate:
    """Evaluate if any predicate is true.

    A predicate is of the form :math:`P\\colon X \\rightarrow \\{False, True\\}`

    Examples
    --------
    >>> from onekit import pytlz
    >>> pytlz.any_predicate_true(lambda x: x % 2 == 0, lambda x: x % 5 == 0)(10)
    True

    >>> is_divisible_by_3_or_5 = pytlz.any_predicate_true(
    ...     pytlz.isdivisibleby(3),
    ...     pytlz.isdivisibleby(5),
    ... )
    >>> type(is_divisible_by_3_or_5)
    <class 'function'>
    >>> is_divisible_by_3_or_5(60)
    True
    >>> is_divisible_by_3_or_5(9)
    True
    >>> is_divisible_by_3_or_5(13)
    False
    """

    def inner(x: Any, /) -> bool:
        """Evaluate all specified predicates :math:`P_i` for value :math:`x \\in X`."""
        return any(predicate(x) for predicate in flatten(predicates))

    return inner


def check_random_state(seed: Seed = None, /) -> random.Random:
    """Turn seed into random.Random instance.

    Examples
    --------
    >>> import random
    >>> from onekit import pytlz
    >>> rng = pytlz.check_random_state()
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
    >>> from onekit import pytlz
    >>> {pytlz.coinflip(0.5) for _ in range(30)} == {True, False}
    True

    >>> fair_coin = partial(pytlz.coinflip, 0.5)
    >>> type(fair_coin)
    <class 'functools.partial'>
    >>> # fix coinflip outcome
    >>> fair_coin(seed=1)  # doctest: +SKIP
    True
    >>> # fix sequence of coinflip outcomes
    >>> rng = pytlz.check_random_state(2)
    >>> [fair_coin(seed=rng) for _ in range(6)]  # doctest: +SKIP
    [False, False, True, True, False, False]

    >>> biased_coin = partial(pytlz.coinflip, 0.6, seed=pytlz.check_random_state(3))
    >>> type(biased_coin)
    <class 'functools.partial'>
    >>> [biased_coin() for _ in range(6)]  # doctest: +SKIP
    [True, True, True, False, False, True]
    """
    if not (0 <= bias <= 1):
        raise ValueError(f"{bias=} - must be a float in [0, 1]")

    rng = check_random_state(seed)

    return rng.random() < bias


def collatz(n: int, /) -> Generator:
    """Generate a Collatz sequence.

    The famous 3n + 1 conjecture [c1]_ [c2]_. Given a positive integer :math:`n > 0`,
    the next term in the Collatz sequence is half of :math:`n`
    if :math:`n` is even; otherwise, if :math:`n` is odd,
    the next term is 3 times :math:`n` plus 1.
    Symbolically,

    .. math::

        f(n) =
        \\begin{cases}
             n / 2 & \\text{ if } n \\equiv 0 \\text{ (mod 2) } \\\\[6pt]
            3n + 1 & \\text{ if } n \\equiv 1 \\text{ (mod 2) }
        \\end{cases}

    The Collatz conjecture is that the sequence always reaches 1
    for any positive integer :math:`n`.

    Parameters
    ----------
    n : int
        A positive integer seeding the Collatz sequence.

    Yields
    ------
    int
        A generator of Collatz numbers that breaks when 1 is reached.

    Raises
    ------
    ValueError
        If ``n`` is not a positive integer.

    References
    ----------
    .. [c1] "Collatz", The On-Line Encyclopedia of Integer Sequences®,
            https://oeis.org/A006370
    .. [c2] "Collatz conjecture", Wikipedia,
            https://en.wikipedia.org/wiki/Collatz_conjecture

    Examples
    --------
    >>> import toolz
    >>> from onekit import pytlz
    >>> n = 12
    >>> list(pytlz.collatz(n))
    [12, 6, 3, 10, 5, 16, 8, 4, 2, 1]
    >>> toolz.count(pytlz.collatz(n))
    10
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"{n=} - must be a positive integer")

    while True:
        yield n

        if n == 1:
            break

        # update
        n = n // 2 if iseven(n) else 3 * n + 1


def contrast_sets(x: set, y: set, /, *, n: int = 3) -> dict:
    """Contrast sets.

    Examples
    --------
    >>> from onekit import pytlz
    >>> a = {"a", "c", "b", "g", "h", "i", "j", "k"}
    >>> b = {"c", "d", "e", "f", "g", "p", "q"}
    >>> summary = pytlz.contrast_sets(a, b)
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


def date_to_str(d: dt.date, /) -> str:
    """Cast date to string in ISO format: YYYY-MM-DD.

    Examples
    --------
    >>> import datetime as dt
    >>> from onekit import pytlz
    >>> pytlz.date_to_str(dt.date(2022, 1, 1))
    '2022-01-01'
    """
    return d.isoformat()


def extend_range(xmin: float, xmax: float, /, *, factor: float = 0.05) -> Pair:
    """Extend value range ``xmax - xmin`` by factor.

    Examples
    --------
    >>> from onekit import pytlz
    >>> pytlz.extend_range(0.0, 1.0)
    (-0.05, 1.05)

    >>> pytlz.extend_range(0.0, 1.0, factor=0.1)
    (-0.1, 1.1)
    """
    if not isinstance(factor, float) or factor < 0:
        raise ValueError(f"{factor=} - must be a non-negative float")

    xmin, xmax = sorted([xmin, xmax])
    value_range = xmax - xmin

    new_xmin = xmin - factor * value_range
    new_xmax = xmax + factor * value_range

    return new_xmin, new_xmax


def fibonacci() -> Generator:
    """Generate the Fibonacci sequence.

    For :math:`n > 1`, Fibonacci numbers may be defined by [f1]_ [f2]_:

    .. math::

        F(n) = F(n-1) + F(n-2) \\text{ with } F(0) = 0 \\text{ and } F(1) = 1.

    As such, the sequence starts as follows:

    .. math::

        0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, \\dots

    Yields
    ------
    int
        A generator of consecutive Fibonacci numbers.

    References
    ----------
    .. [f1] "Fibonacci numbers", The On-Line Encyclopedia of Integer Sequences®,
            https://oeis.org/A000045
    .. [f2] "Fibonacci number", Wikipedia,
            https://en.wikipedia.org/wiki/Fibonacci_number

    Examples
    --------
    >>> import toolz
    >>> from onekit import pytlz
    >>> list(toolz.take(13, pytlz.fibonacci()))
    [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    """
    lag2, lag1 = 0, 1
    yield lag2
    yield lag1

    while True:
        lag0 = lag2 + lag1
        yield lag0
        lag2, lag1 = lag1, lag0


def flatten(*items: Sequence[Any]) -> Generator:
    """Flatten sequence of items.

    Examples
    --------
    >>> from onekit import pytlz
    >>> list(pytlz.flatten([[1, 2], *[3, 4], [5]]))
    [1, 2, 3, 4, 5]

    >>> list(pytlz.flatten([1, (2, 3)], 4, [], [[[5]], 6]))
    [1, 2, 3, 4, 5, 6]

    >>> list(pytlz.flatten(["one", 2], 3, [(4, "five")], [[["six"]]], "seven", []))
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
    >>> from onekit import pytlz
    >>> def foobar():
    ...     return pytlz.func_name()
    ...
    >>> foobar()
    'foobar'
    """
    return inspect.stack()[1].function


@toolz.curry
def isdivisibleby(n: int, x: Union[int, float], /) -> bool:
    """Evaluate if :math:`x` is evenly divisible by :math:`n`.

    Examples
    --------
    >>> from onekit import pytlz
    >>> pytlz.isdivisibleby(7, 49)
    True

    >>> # function is curried
    >>> pytlz.isdivisibleby(5)(10)
    True
    >>> is_divisible_by_5 = pytlz.isdivisibleby(5)
    >>> is_divisible_by_5(10)
    True
    >>> is_divisible_by_5(11.0)
    False
    """
    return x % n == 0


def iseven(x: Union[int, float], /) -> bool:
    """Evaluate if :math:`x` is even.

    Examples
    --------
    >>> from onekit import pytlz
    >>> pytlz.iseven(0)
    True

    >>> pytlz.iseven(1)
    False

    >>> pytlz.iseven(2)
    True
    """
    return isdivisibleby(2)(x)


def isodd(x: Union[int, float], /) -> bool:
    """Evaluate if :math:`x` is odd.

    Examples
    --------
    >>> from onekit import pytlz
    >>> pytlz.isodd(0)
    False

    >>> pytlz.isodd(1)
    True

    >>> pytlz.isodd(2)
    False
    """
    return toolz.complement(iseven)(x)


def num_to_str(x: Union[int, float], /) -> str:
    """Cast number to string with underscores as thousands separator.

    Examples
    --------
    >>> from onekit import pytlz
    >>> pytlz.num_to_str(1000000)
    '1_000_000'

    >>> pytlz.num_to_str(100000.0)
    '100_000.0'
    """
    return f"{x:_}"


@toolz.curry
def reduce_sets(func: Callable[[set, set], set], /, *sets: Sequence[set]) -> set:
    """Apply function of two set arguments to reduce a sequence of sets.

    Examples
    --------
    >>> from onekit import pytlz
    >>> x = {0, 1, 2, 3}
    >>> y = {2, 4, 6}
    >>> z = {2, 6, 8}
    >>> pytlz.reduce_sets(set.intersection, x, y, z)
    {2}
    >>> sets = [x, y, z]
    >>> pytlz.reduce_sets(set.symmetric_difference, sets)
    {0, 1, 2, 3, 4, 8}
    >>> pytlz.reduce_sets(set.difference, *sets)
    {0, 1, 3}

    >>> # function is curried
    >>> pytlz.reduce_sets(set.union)(x, y, z)
    {0, 1, 2, 3, 4, 6, 8}
    >>> pytlz.reduce_sets(set.union)(sets)
    {0, 1, 2, 3, 4, 6, 8}
    >>> set_union = pytlz.reduce_sets(set.union)
    >>> set_union(*sets)
    {0, 1, 2, 3, 4, 6, 8}
    """
    return toolz.pipe(sets, flatten, map(set), reduce(func))


def signif(x: Union[int, float], /, *, n: int = 3) -> Union[int, float]:
    """Round :math:`x` to its :math:`n` significant digits.

    Examples
    --------
    >>> from onekit import pytlz
    >>> pytlz.signif(987654321)
    988000000

    >>> pytlz.signif(14393237.76, n=2)
    14000000.0

    >>> pytlz.signif(14393237.76, n=3)
    14400000.0
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"{n=} - must be a positive integer")

    if not math.isfinite(x) or math.isclose(x, 0.0):
        return x

    n -= math.ceil(math.log10(abs(x)))
    return round(x, n)


def source_code(x: object, /) -> str:
    """Get source code of an object.

    Examples
    --------
    >>> from onekit import pytlz
    >>> def greet():
    ...     return "Hello, World!"
    ...
    >>> print(pytlz.source_code(greet))
    def greet():
        return "Hello, World!"
    <BLANKLINE>
    """
    return inspect.getsource(x)
