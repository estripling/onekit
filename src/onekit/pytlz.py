"""Python toolz."""

import datetime as dt
import inspect
from typing import (
    Any,
    Callable,
    Generator,
    Iterator,
    List,
    Sequence,
    Tuple,
    Union,
)

import toolz

__all__ = (
    "all_predicate_true",
    "any_predicate_true",
    "date_to_str",
    "extend_range",
    "flatten",
    "func_name",
    "isdivisibleby",
    "iseven",
    "isodd",
    "num_to_str",
    "source_code",
)


Pair = Tuple[float, float]


@toolz.curry
def all_predicate_true(predicates: List[Callable[[Any], bool]], x: Any, /) -> bool:
    """Check if all predicates are true.

    Examples
    --------
    >>> from onekit import pytlz
    >>> is_divisible_by_3_and_5 = pytlz.all_predicate_true(
    ...     [
    ...         pytlz.isdivisibleby(3),
    ...         pytlz.isdivisibleby(5),
    ...     ]
    ... )
    >>> is_divisible_by_3_and_5(60)
    True
    >>> is_divisible_by_3_and_5(9)
    False
    """
    return all(predicate(x) for predicate in predicates)


@toolz.curry
def any_predicate_true(predicates: List[Callable[[Any], bool]], x: Any, /) -> bool:
    """Check if any predicate is true.

    Examples
    --------
    >>> from onekit import pytlz
    >>> is_divisible_by_3_and_5 = pytlz.any_predicate_true(
    ...     [
    ...         pytlz.isdivisibleby(3),
    ...         pytlz.isdivisibleby(5),
    ...     ]
    ... )
    >>> is_divisible_by_3_and_5(60)
    True
    >>> is_divisible_by_3_and_5(9)
    True
    >>> is_divisible_by_3_and_5(13)
    False
    """
    return any(predicate(x) for predicate in predicates)


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


@toolz.curry
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
    """Check if :math:`x` is evenly divisible by :math:`n`.

    Examples
    --------
    >>> from onekit import pytlz
    >>> is_div_5 = pytlz.isdivisibleby(5)
    >>> is_div_5(10)
    True
    >>> is_div_5(11.0)
    False
    """
    return x % n == 0


def iseven(x: Union[int, float], /) -> bool:
    """Check if :math:`x` is even.

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
    """Check if :math:`x` is odd.

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


def num_to_str(n: Union[int, float], /) -> str:
    """Cast number to string with underscores as thousands separator.

    Examples
    --------
    >>> from onekit import pytlz
    >>> pytlz.num_to_str(1000000)
    '1_000_000'

    >>> pytlz.num_to_str(100000.0)
    '100_000.0'
    """
    return f"{n:_}"


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
