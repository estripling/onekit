import datetime as dt
from typing import Union

__all__ = (
    "date_to_str",
    "num_to_str",
)


def date_to_str(d: dt.date, /) -> str:
    """Cast date to string in ISO format: YYYY-MM-DD.

    Examples
    --------
    >>> import onekit as ok
    >>> import datetime as dt
    >>> ok.date_to_str(dt.date(2022, 1, 1))
    '2022-01-01'
    """
    return d.isoformat()


def num_to_str(n: Union[int, float], /) -> str:
    """Cast number to string with underscores as thousands separator.

    Examples
    --------
    >>> import onekit as ok
    >>> ok.num_to_str(1000000)
    '1_000_000'

    >>> ok.num_to_str(100000.0)
    '100_000.0'
    """
    return f"{n:_}"
