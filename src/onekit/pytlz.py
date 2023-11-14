"""Python toolz."""

import datetime as dt
from typing import (
    Any,
    Generator,
    Iterator,
    Sequence,
    Union,
)

__all__ = (
    "date_to_str",
    "flatten",
    "num_to_str",
)


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
