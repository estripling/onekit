import calendar
import datetime as dt
import itertools
import math
import operator
from contextlib import ContextDecorator
from typing import (
    Generator,
    NamedTuple,
)

import pytz
import toolz
from dateutil.relativedelta import relativedelta

from onekit import pythonkit as pk

__all__ = (
    "DateRange",
    "ElapsedTime",
    "create_date_range",
    "date_ago",
    "date_ahead",
    "date_count_backward",
    "date_count_forward",
    "date_diff",
    "date_range",
    "date_to_str",
    "humantime",
    "last_date_of_month",
    "num_days",
    "stopwatch",
    "str_to_date",
    "timestamp",
    "weekday",
)

from onekit.exception import InvalidChoiceError


class ElapsedTime(NamedTuple):
    """Represents an immutable, human-readable format for elapsed time.

    See Also
    --------
    date_diff : Function to compute the date difference between two dates.

    Examples
    --------
    >>> from onekit.timekit import ElapsedTime
    >>> et = ElapsedTime(years=0, months=0, weeks=0, days=6)
    >>> et
    ElapsedTime(years=0, months=0, weeks=0, days=6)
    >>> str(et)
    '0y 0m 0w 6d'
    """

    years: int
    months: int
    weeks: int
    days: int

    def __str__(self) -> str:
        return pk.concat_strings(
            " ",
            f"{self.years}y",
            f"{self.months}m",
            f"{self.weeks}w",
            f"{self.days}d",
        )


class DateRange(NamedTuple):
    """Represents an immutable range of dates, defined by its minimum and maximum date.

    Notes
    -----
    Validation and ordering (e.g., ensuring `min_date <= max_date`)
    are NOT performed automatically by the `NamedTuple`'s constructor.
    It is highly recommended to use the factory function (e.g., `create_date_range`)
    to instantiate `DateRange` objects robustly, ensuring proper order
    and any other custom initialization logic.

    See Also
    --------
    create_date_range : Factory function for robust instantiation.

    Examples
    --------
    >>> import datetime as dt
    >>> from onekit.timekit import DateRange
    >>> dr = DateRange(dt.date(2025, 6, 1), dt.date(2025, 6, 3))
    >>> dr
    DateRange(min_date=datetime.date(2025, 6, 1), max_date=datetime.date(2025, 6, 3))

    >>> str(dr)
    'DateRange(2025-06-01, 2025-06-03) - 3 days in total - elapsed time 0y 0m 0w 2d'

    >>> dr.elapsed_time
    ElapsedTime(years=0, months=0, weeks=0, days=2)

    >>> dr.elapsed_days
    2

    >>> dr.elapsed_years
    0

    >>> dr.number_of_days
    3

    >>> list(dr.make_date_range())
    [datetime.date(2025, 6, 1), datetime.date(2025, 6, 2), datetime.date(2025, 6, 3)]
    """

    min_date: dt.date
    max_date: dt.date

    def __str__(self) -> str:
        n = self.number_of_days
        return (
            "{cls_name}({min_date}, {max_date}) - {n} in total - elapsed time {x}"
        ).format(
            cls_name=self.__class__.__name__,
            min_date=date_to_str(self.min_date),
            max_date=date_to_str(self.max_date),
            n="1 day" if n == 1 else f"{pk.num_to_str(n)} days",
            x=str(self.elapsed_time),
        )

    @property
    def elapsed_time(self) -> ElapsedTime:
        """Compute the elapsed time between the min and max date (max exclusive)."""
        return date_diff(self.min_date, self.max_date, unit=None)

    @property
    def elapsed_days(self) -> int:
        """Compute the elapsed days between the min and max date (max exclusive)."""
        return date_diff(self.min_date, self.max_date, unit="days")

    @property
    def elapsed_years(self) -> int:
        """Compute the elapsed years between the min and max date (max exclusive)."""
        return date_diff(self.min_date, self.max_date, unit="years")

    @property
    def number_of_days(self) -> int:
        """Compute the number of days between the min and max date (both inclusive)."""
        return num_days(self.min_date, self.max_date)

    def make_date_range(
        self,
        *,
        incl_min: bool = True,
        incl_max: bool = True,
    ) -> Generator:
        """Generate sequence of consecutive dates between the min and max date."""
        return date_range(
            self.min_date,
            self.max_date,
            incl_min=incl_min,
            incl_max=incl_max,
        )


def create_date_range(min_date: dt.date, max_date: dt.date, /) -> DateRange:
    """Creates a DateRange NamedTuple.

    This is a factory function for creating `DateRange` instances with automatic date
    ordering, i.e., it ensures `min_date` is always less than or equal to `max_date`.
    If `min_date` > `max_date`, they are swapped.

    Parameters
    ----------
    min_date : datetime.date
        The chronological earliest date in the range.
    max_date : datetime.date
        The chronological latest date in the range.

    Returns
    -------
    DateRange
        An immutable date range object with `min_date <= max_date`.

    See Also
    --------
    DateRange : The immutable date range object returned by this factory.

    Examples
    --------
    >>> import datetime as dt
    >>> from onekit import timekit as tk
    >>> # standard usage
    >>> tk.create_date_range(dt.date(2025, 6, 1), dt.date(2025, 6, 7))
    DateRange(min_date=datetime.date(2025, 6, 1), max_date=datetime.date(2025, 6, 7))

    >>> # single day range
    >>> tk.create_date_range(dt.date(2025, 6, 1), dt.date(2025, 6, 1))
    DateRange(min_date=datetime.date(2025, 6, 1), max_date=datetime.date(2025, 6, 1))

    >>> # dates provided in reverse order are automatically swapped
    >>> tk.create_date_range(dt.date(2025, 6, 7), dt.date(2025, 6, 1))
    DateRange(min_date=datetime.date(2025, 6, 1), max_date=datetime.date(2025, 6, 7))
    """
    if min_date > max_date:
        min_date, max_date = max_date, min_date
    return DateRange(min_date, max_date)


def date_ago(ref_date: dt.date, /, n: int) -> dt.date:
    """Compute date :math:`n \\in \\mathbb{N}_{0}` days ago from reference date.

    Examples
    --------
    >>> import datetime as dt
    >>> from onekit import timekit as tk
    >>> ref_date = dt.date(2022, 1, 1)
    >>> tk.date_ago(ref_date, n=0)
    datetime.date(2022, 1, 1)
    >>> tk.date_ago(ref_date, n=1)
    datetime.date(2021, 12, 31)
    >>> tk.date_ago(ref_date, n=2)
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
    >>> from onekit import timekit as tk
    >>> ref_date = dt.date(2022, 1, 1)
    >>> tk.date_ahead(ref_date, n=0)
    datetime.date(2022, 1, 1)
    >>> tk.date_ahead(ref_date, n=1)
    datetime.date(2022, 1, 2)
    >>> tk.date_ahead(ref_date, n=2)
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
    >>> from onekit import timekit as tk
    >>> ref_date = dt.date(2022, 1, 1)
    >>> curried.pipe(
    ...     tk.date_count_backward(ref_date),
    ...     curried.map(tk.date_to_str),
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
    >>> from onekit import timekit as tk
    >>> ref_date = dt.date(2022, 1, 1)
    >>> curried.pipe(
    ...     tk.date_count_forward(ref_date),
    ...     curried.map(tk.date_to_str),
    ...     curried.take(3),
    ...     list,
    ... )
    ['2022-01-01', '2022-01-02', '2022-01-03']
    """
    successor = operator.add
    return toolz.iterate(lambda d: successor(d, dt.timedelta(1)), ref_date)


# noinspection PyUnreachableCode
def date_diff(
    min_date: dt.date,
    max_date: dt.date,
    /,
    *,
    unit: str | None = None,
) -> ElapsedTime | int:
    """Compute date difference between the min and max date (max exclusive).

    See Also
    --------
    ElapsedTime : The immutable, human-readable object for elapsed time.

    Examples
    --------
    >>> import datetime as dt
    >>> from onekit import timekit as tk
    >>> d1 = dt.date(2024, 7, 1)
    >>> d2 = dt.date(2024, 7, 7)

    >>> dd = tk.date_diff(d1, d2, unit=None)
    >>> dd
    ElapsedTime(years=0, months=0, weeks=0, days=6)
    >>> str(dd)
    '0y 0m 0w 6d'

    >>> tk.date_diff(d1, d1, unit="days")
    0

    >>> tk.date_diff(d1, d2, unit="days")
    6

    >>> tk.date_diff(d2, d1, unit="days")
    -6

    >>> tk.date_diff(d1, d2, unit="years")
    0
    """
    choices = [None, "days", "years"]
    match unit:
        case None:
            delta = relativedelta(max_date, min_date)
            return ElapsedTime(
                years=delta.years,
                months=delta.months,
                weeks=delta.days // 7,
                days=delta.days % 7,
            )

        case "days":
            return (max_date - min_date).days

        case "years":
            return relativedelta(max_date, min_date).years

        case _:
            raise InvalidChoiceError(unit, choices)


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
    >>> from onekit import timekit as tk
    >>> d1 = dt.date(2022, 1, 1)
    >>> d2 = dt.date(2022, 1, 3)

    >>> curried.pipe(tk.date_range(d1, d2), curried.map(tk.date_to_str), list)
    ['2022-01-01', '2022-01-02', '2022-01-03']

    >>> curried.pipe(
    ...     tk.date_range(d1, d2, incl_min=False, incl_max=True),
    ...     curried.map(tk.date_to_str),
    ...     list,
    ... )
    ['2022-01-02', '2022-01-03']

    >>> curried.pipe(
    ...     tk.date_range(d1, d2, incl_min=True, incl_max=False),
    ...     curried.map(tk.date_to_str),
    ...     list,
    ... )
    ['2022-01-01', '2022-01-02']

    >>> curried.pipe(
    ...     tk.date_range(d1, d2, incl_min=False, incl_max=False),
    ...     curried.map(tk.date_to_str),
    ...     list,
    ... )
    ['2022-01-02']

    >>> list(tk.date_range(d1, dt.date(2022, 1, 1)))
    [datetime.date(2022, 1, 1)]

    >>> list(tk.date_range(d1, dt.date(2022, 1, 1), incl_min=False))
    []

    >>> # function makes sure: start <= end
    >>> curried.pipe(tk.date_range(d2, d1), curried.map(tk.date_to_str), list)
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
    >>> from onekit import timekit as tk
    >>> tk.date_to_str(dt.date(2022, 1, 1))
    '2022-01-01'
    """
    return d.isoformat()


def humantime(seconds: int | float, /) -> str:
    """Convert seconds to human-readable time.

    Examples
    --------
    >>> from onekit import timekit as tk
    >>> # 1 second
    >>> tk.humantime(1)
    '1s'

    >>> # 1 minute
    >>> tk.humantime(60)
    '1m'

    >>> # 1 hour
    >>> tk.humantime(60 * 60)
    '1h'

    >>> # 1 day
    >>> tk.humantime(60 * 60 * 24)
    '1d'

    >>> tk.humantime(60 * 60 * 24 + 60 * 60 + 60 + 1)
    '1d 1h 1m 1s'

    >>> tk.humantime(3 * 60 * 60 * 24 + 2 * 60)
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
    >>> from onekit import timekit as tk
    >>> tk.last_date_of_month(2022, 1)
    datetime.date(2022, 1, 31)
    """
    _, number_of_days_in_month = calendar.monthrange(year, month)
    return dt.date(year, month, number_of_days_in_month)


def num_days(min_date: dt.date, max_date: dt.date, /) -> int:
    """Compute the number of days between two dates (both endpoints inclusive).

    Examples
    --------
    >>> import datetime as dt
    >>> from onekit import timekit as tk
    >>> tk.num_days(dt.date(2022, 8, 1), dt.date(2022, 8, 1))
    1

    >>> tk.num_days(dt.date(2022, 8, 1), dt.date(2022, 8, 2))
    2

    >>> tk.num_days(dt.date(2022, 8, 1), dt.date(2022, 8, 7))
    7

    >>> # function makes sure: start <= end
    >>> tk.num_days(dt.date(2022, 8, 7), dt.date(2022, 8, 1))
    7
    """
    return abs(date_diff(min_date, max_date, unit="days")) + 1


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
    >>> from onekit import timekit as tk
    >>> with tk.stopwatch():  # doctest: +SKIP
    ...     time.sleep(0.1)
    ...
    2023-01-01 12:00:00 -> 2023-01-01 12:00:00 = 0.100691s

    >>> # as decorator
    >>> import time
    >>> from onekit import timekit as tk
    >>> @tk.stopwatch()
    ... def func():
    ...     time.sleep(0.1)
    ...
    >>> func()  # doctest: +SKIP
    2023-01-01 12:00:00 -> 2023-01-01 12:00:00 = 0.100709s - func

    >>> # stopwatch instance
    >>> import time
    >>> from onekit import timekit as tk
    >>> with tk.stopwatch("instance-example") as sw:  # doctest: +SKIP
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
    >>> from onekit import timekit as tk
    >>> with tk.stopwatch(1) as sw1:  # doctest: +SKIP
    ...     time.sleep(1)
    ...
    2023-01-01 12:00:00 -> 2023-01-01 12:00:01 = 1.00122s - 1
    >>> with tk.stopwatch(2) as sw2:  # doctest: +SKIP
    ...     time.sleep(1)
    ...
    2023-01-01 12:01:00 -> 2023-01-01 12:01:01 = 1.00121s - 2
    >>> with tk.stopwatch(3) as sw3:  # doctest: +SKIP
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


def str_to_date(s: str, /) -> dt.date:
    """Cast ISO date string to date.

    Examples
    --------
    >>> from onekit import timekit as tk
    >>> tk.str_to_date("2022-01-01")
    datetime.date(2022, 1, 1)
    """
    return dt.datetime.strptime(s, "%Y-%m-%d").date()


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
    >>> from onekit import timekit as tk
    >>> tk.timestamp()  # doctest: +SKIP
    '2024-01-01 00:00:00'

    >>> tk.timestamp("CET")  # doctest: +SKIP
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
    >>> from onekit import timekit as tk
    >>> tk.weekday(dt.date(2022, 8, 1))
    'Mon'
    >>> tk.weekday(dt.date(2022, 8, 7))
    'Sun'
    """
    return d.strftime("%a")
