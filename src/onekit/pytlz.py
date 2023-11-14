"""Python toolz."""

import datetime as dt
from typing import (
    Any,
    Callable,
    Generator,
    Iterator,
    List,
    Sequence,
    Union,
)

import toolz

__all__ = (
    "all_predicate_true",
    "date_to_str",
    "flatten",
    "isdivisibleby",
    "iseven",
    "isodd",
    "num_to_str",
)


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
