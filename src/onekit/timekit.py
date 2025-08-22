import datetime as dt
import functools
import math
from contextlib import ContextDecorator
from typing import Callable
from zoneinfo import (
    ZoneInfo,
    ZoneInfoNotFoundError,
    available_timezones,
)

import duckdb
from dateutil import parser
from dateutil.relativedelta import relativedelta

from onekit import pythonkit as pk
from onekit.exception import InvalidChoiceError

__all__ = (
    "Duration",
    "datesub",
    "stopwatch",
    "timestamp",
    "to_datetime",
)

DateTimeLike = dt.datetime | dt.date | dt.time | str


class Duration:
    """Immutable duration object representing the difference between two dates/times.

    Examples
    --------
    >>> from onekit import timekit as tk
    >>> duration = tk.Duration("2024-07-01 13:00:00", "2024-08-02 14:00:01")
    >>> duration
    Duration(2024-07-01T13:00:00, 2024-08-02T14:00:01)
    >>> duration.is_zero
    False
    >>> duration.as_default()
    '1mo 1d 1h 1s'
    >>> duration.as_default() == str(duration)
    True
    >>> duration.as_compact_days()
    '32d 1h 1s'
    >>> duration.as_compact_weeks()
    '1mo 1d 1h 1s'
    >>> duration.as_iso()
    'P1M1DT1H1S'
    >>> duration.as_total_seconds()
    '2_768_401s'
    """

    __slots__ = ("_start_dt", "_end_dt", "_delta")

    def __init__(self, start: DateTimeLike, end: DateTimeLike, /):
        start_dt = to_datetime(start)
        end_dt = to_datetime(end)

        if end_dt < start_dt:
            start_dt, end_dt = end_dt, start_dt

        self._start_dt = start_dt
        self._end_dt = end_dt
        self._delta = relativedelta(end_dt, start_dt)

    @property
    def start_dt(self) -> dt.datetime:
        """Return the start datetime."""
        return self._start_dt

    @property
    def end_dt(self) -> dt.datetime:
        """Return the end datetime."""
        return self._end_dt

    @property
    def delta(self) -> relativedelta:
        """Return the relativedelta object."""
        return self._delta

    @property
    def years(self) -> int:
        """Return the number of whole years between start and end datetime values."""
        return self.delta.years

    @property
    def months(self) -> int:
        """Return the number of whole months (excluding years)."""
        return self.delta.months

    @property
    def days(self) -> int:
        """Return the number of days (excluding months and years)."""
        return self.delta.days

    @property
    def hours(self) -> int:
        """Return the number of hours (excluding days)."""
        return self.delta.hours

    @property
    def minutes(self) -> int:
        """Return the number of minutes (excluding hours)."""
        return self.delta.minutes

    @property
    def seconds(self) -> float:
        """Return the remaining seconds (excluding minutes)."""
        return self.delta.seconds

    @property
    def microseconds(self) -> int:
        """Return the number of microseconds (excluding seconds)."""
        return self.delta.microseconds

    @property
    def total_seconds(self) -> float:
        """Return the total duration in seconds."""
        return (self.end_dt - self.start_dt).total_seconds()

    @property
    def is_zero(self) -> bool:
        """Return True if the duration is zero (all parts are zero).

        Examples
        --------
        >>> from onekit import timekit as tk
        >>> duration = tk.Duration("13:00:00", "13:00:00")
        >>> duration.is_zero
        True
        """
        parts = (
            self.years,
            self.months,
            self.days,
            self.hours,
            self.minutes,
            self.seconds,
        )
        return all(v == 0 for v in parts) and math.isclose(self.microseconds, 0)

    @property
    def formatted_seconds(self) -> str:
        """Return seconds and microseconds as a formatted string.

        Examples
        --------
        >>> from onekit import timekit as tk
        >>> duration = tk.Duration("13:00:00", "13:00:00.123456")
        >>> duration.seconds, duration.microseconds, duration.formatted_seconds
        (0, 123456, '0.123456')

        >>> duration = tk.Duration("13:00:00", "13:00:01")
        >>> duration.seconds, duration.microseconds, duration.formatted_seconds
        (1, 0, '1')

        >>> duration = tk.Duration("13:00:00", "13:00:01.234")
        >>> duration.seconds, duration.microseconds, duration.formatted_seconds
        (1, 234000, '1.234')
        """
        return (
            f"{self.seconds:d}.{self.microseconds:06d}".rstrip("0")
            if self.microseconds
            else f"{self.seconds:d}" if self.seconds else "0"
        )

    def as_default(self) -> str:
        """Return duration as a human-readable string.

        Examples
        --------
        >>> from onekit import timekit as tk
        >>> duration = tk.Duration("2024-07-01T13:00:00", "2025-07-01T14:01:01")
        >>> duration.as_default()
        '1y 1h 1m 1s'

        >>> duration = tk.Duration("2024-07-01T13:00:00", "2025-07-09T14:01:01")
        >>> duration.as_default()
        '1y 8d 1h 1m 1s'
        """
        parts = []
        if self.years:
            parts.append(f"{self.years}y")
        if self.months:
            parts.append(f"{self.months}mo")
        if self.days:
            parts.append(f"{self.days}d")
        if self.hours:
            parts.append(f"{self.hours}h")
        if self.minutes:
            parts.append(f"{self.minutes}m")

        seconds_part = self._get_seconds_part(num_parts=len(parts))
        if seconds_part:
            parts.append(seconds_part)

        return " ".join(parts)

    def as_compact_days(self) -> str:
        """Return duration as a compact human-readable string with days as largest unit.

        Examples
        --------
        >>> from onekit import timekit as tk
        >>> duration = tk.Duration("2024-07-01T13:00:00", "2025-07-01T14:01:01")
        >>> duration.as_compact_days()
        '365d 1h 1m 1s'

        >>> duration = tk.Duration("2024-07-01T13:00:00", "2025-07-09T14:01:01")
        >>> duration.as_compact_days()
        '373d 1h 1m 1s'
        """
        minutes, seconds = divmod(int(round(self.total_seconds, 0)), 60)
        hours, minutes = divmod(minutes, 60)
        days, hours = divmod(hours, 24)

        parts = []
        if days:
            parts.append(f"{days}d")
        if hours:
            parts.append(f"{hours}h")
        if minutes:
            parts.append(f"{minutes}m")

        seconds_part = self._get_seconds_part(num_parts=len(parts))
        if seconds_part:
            parts.append(seconds_part)

        return " ".join(parts)

    def as_compact_weeks(self) -> str:
        """Return duration as a compact human-readable string including weeks.

        Examples
        --------
        >>> from onekit import timekit as tk
        >>> duration = tk.Duration("2024-07-01T13:00:00", "2025-07-01T14:01:01")
        >>> duration.as_compact_weeks()
        '1y 1h 1m 1s'

        >>> duration = tk.Duration("2024-07-01T13:00:00", "2025-07-09T14:01:01")
        >>> duration.as_compact_weeks()
        '1y 1w 1d 1h 1m 1s'
        """
        weeks = self.delta.days // 7
        days = self.delta.days % 7

        parts = []
        if self.years:
            parts.append(f"{self.years}y")
        if self.months:
            parts.append(f"{self.months}mo")
        if weeks:
            parts.append(f"{weeks}w")
        if days:
            parts.append(f"{days}d")
        if self.hours:
            parts.append(f"{self.hours}h")
        if self.minutes:
            parts.append(f"{self.minutes}m")

        seconds_part = self._get_seconds_part(num_parts=len(parts))
        if seconds_part:
            parts.append(seconds_part)

        return " ".join(parts)

    def as_iso(self) -> str:
        """Return duration as an ISO 8601 duration string.

        Examples
        --------
        >>> from onekit import timekit as tk
        >>> duration = tk.Duration("2024-07-01T13:00:00", "2025-07-01T14:01:01")
        >>> duration.as_iso()
        'P1YT1H1M1S'

        >>> duration = tk.Duration("2024-07-01T13:00:00", "2025-07-09T14:01:01")
        >>> duration.as_iso()
        'P1Y8DT1H1M1S'
        """
        date_parts = []
        if self.years:
            date_parts.append(f"{self.years}Y")
        if self.months:
            date_parts.append(f"{self.months}M")
        if self.days:
            date_parts.append(f"{self.days}D")

        time_parts = []
        if self.hours:
            time_parts.append(f"{self.hours}H")
        if self.minutes:
            time_parts.append(f"{self.minutes}M")

        seconds_part = self._get_seconds_part(num_parts=len(time_parts), unit="S")
        if seconds_part != "0S":
            time_parts.append(seconds_part)

        if len(date_parts) == 0 and len(time_parts) == 0:
            time_parts.append("0S")

        parts = ["P"]
        if date_parts:
            parts.extend(date_parts)
        if time_parts:
            parts.append("T")
            parts.extend(time_parts)

        return "".join(parts)

    def as_total_seconds(self) -> str:
        """Return the total duration in seconds as a string.

        Examples
        --------
        >>> from onekit import timekit as tk
        >>> duration = tk.Duration("2024-07-01T13:00:00", "2025-07-01T14:01:01")
        >>> duration.as_total_seconds()
        '31_539_661s'
        """
        return f"{pk.num_to_str(self.total_seconds)}s"

    def as_custom(self, formatter: Callable[["Duration"], str]) -> str:
        """Return a custom string representation of the Duration object.

        Examples
        --------
        >>> from onekit import timekit as tk
        >>> duration = tk.Duration("2024-07-01T13:00:00", "2025-07-01T14:01:01")
        >>> duration.as_custom(lambda x: f"{x.years}y {x.months}mo {x.days}d")
        '1y 0mo 0d'
        """
        return formatter(self)

    def _get_seconds_part(self, num_parts: int, unit: str = "s") -> str:
        """Helper function to process the seconds part."""
        fmt_secs = self.formatted_seconds
        return (
            f"{fmt_secs}{unit}"
            if fmt_secs != "0"
            else f"0{unit}" if num_parts == 0 else ""
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"({self.start_dt.isoformat()}, {self.end_dt.isoformat()})"
        )

    def __str__(self) -> str:
        return self.as_default()


def datesub(part: str, start: DateTimeLike, end: DateTimeLike, /) -> int:
    """Return the number of complete partitions between times.

    Function computes the difference of fully elapsed time units between the start and
    end date/time values using DuckDB's datesub time function for date subtraction.
    [1]_ [2]_

    References
    ----------
    .. [1] "Time Functions: datesub", DuckDB Documentation,
           `<https://duckdb.org/docs/stable/sql/functions/time>`_
    .. [2] "Date Part Functions", DuckDB Documentation,
            `<https://duckdb.org/docs/stable/sql/functions/datepart.html>`_

    Examples
    --------
    >>> from onekit import timekit as tk
    >>> start_time = "2000-01-01 00:00:00.012345"
    >>> end_time = "2025-01-01 23:59:59.123456"
    >>> tk.datesub("decades", start_time, end_time)
    2
    >>> tk.datesub("years", start_time, end_time)
    25
    >>> tk.datesub("quarter", start_time, end_time)
    100
    >>> tk.datesub("months", start_time, end_time)
    300
    >>> tk.datesub("days", start_time, end_time)
    9132
    >>> tk.datesub("hours", start_time, end_time)
    219191
    >>> tk.datesub("minutes", start_time, end_time)
    13151519
    >>> tk.datesub("seconds", start_time, end_time)
    789091199

    >>> tk.datesub("days", "2024-07-01", "2024-07-07")
    6
    """
    params = (to_datetime(start), to_datetime(end))
    return duckdb.execute(f"SELECT datesub('{part}', ?, ?)", params).fetchone()[0]


class stopwatch(ContextDecorator):
    """Measure elapsed wall-clock time.

    Parameters
    ----------
    label : str, int, optional
        Specify label. If used as a decorator and label is not specified,
        the label is the name of the function.
    timezone : str, optional
        Specify timezone. Default: local timezone.
    fmt : str, optional
        Specify timestamp format. Default: ISO format.

    Notes
    -----
    - Instantiation and use of an instance's properties is only possible
      when ``stopwatch`` is used as a context manager (see examples).

    Examples
    --------
    >>> # as context manager
    >>> import time
    >>> from onekit import timekit as tk
    >>> with tk.stopwatch():  # doctest: +SKIP
    ...     time.sleep(0.1)
    ...
    2024-01-01 13:00:00+00:00 -> 2024-01-01 13:00:00.123456+00:00 took 0.123456s

    >>> # as decorator
    >>> import time
    >>> from onekit import timekit as tk
    >>> @tk.stopwatch()
    ... def func():
    ...     time.sleep(0.1)
    ...
    >>> func()  # doctest: +SKIP
    2024-01-01 13:00:00+00:00 -> 2024-01-01 13:00:00.123456+00:00 took 0.123456s - func

    >>> # stopwatch instance
    >>> import time
    >>> from onekit import timekit as tk
    >>> with tk.stopwatch("test") as sw:  # doctest: +SKIP
    ...     time.sleep(0.1)
    ...
    2024-01-01 13:00:00+00:00 -> 2024-01-01 13:00:00.123456+00:00 took 0.123456s - test
    >>> sw.label  # doctest: +SKIP
    'test'
    >>> sw.fmt is None  # doctest: +SKIP
    True
    >>> sw.start  # doctest: +SKIP
    '2024-01-01 13:00:00+00:00'
    >>> sw.end  # doctest: +SKIP
    '2024-01-01 13:00:00.123456+00:00'
    >>> sw.duration  # doctest: +SKIP
    Duration(2024-01-01 13:00:00+00:00, 2024-01-01 13:00:00.123456+00:00)
    """

    __slots__ = (
        "_label",
        "_timezone",
        "_fmt",
        "_start",
        "_end",
        "_duration",
        "_final_time",
    )

    def __init__(
        self,
        label: str | int | None = None,
        timezone: str | None = None,
        fmt: str | None = None,
    ):
        self._label = label
        self._timezone = timezone
        self._fmt = fmt
        self._start = None
        self._end = None
        self._duration = None
        self._final_time = None

    @property
    def label(self) -> str | int | None:
        """Return the label."""
        return self._label

    @property
    def timezone(self) -> str | None:
        """Return the timezone."""
        return self._timezone

    @property
    def fmt(self) -> str | None:
        """Return the timestamp format."""
        return self._fmt

    @property
    def start(self) -> str:
        """Return the start timestamp."""
        return self._start

    @property
    def end(self) -> str:
        """Return the end timestamp."""
        return self._end

    @property
    def duration(self) -> Duration:
        """Return the Duration object."""
        return self._duration

    @property
    def final_time(self) -> str:
        """Return the final stopwatch time as string."""
        return self._final_time

    def __call__(self, func):
        if self._label is None:
            self._label = func.__name__
        return super().__call__(func)

    def __enter__(self):
        self._start = timestamp(self._timezone, fmt=self._fmt)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._end = timestamp(self._timezone, fmt=self._fmt)
        to_dt = functools.partial(to_datetime, fmt=self.fmt)
        self._duration = Duration(to_dt(self.start), to_dt(self.end))
        self._set_final_time()
        print(self.final_time)
        return False

    def _set_final_time(self) -> None:
        if self._final_time is None:
            self._final_time = pk.concat_strings(
                "",
                f"{self.start} -> {self.end} took {str(self.duration)}",
                f" - {self.label}" if self.label is not None else None,
            )

    def __repr__(self):
        return super().__repr__() if self.final_time is None else self.final_time


def timestamp(timezone: str | None = None, /, *, fmt: str | None = None) -> str:
    """Return timestamp with timezone information.

    Parameters
    ----------
    timezone : str, optional
        Specify timezone. Default: local timezone.
    fmt : str, optional
        Specify timestamp format. Default: ISO format.

    Raises
    ------
    InvalidChoiceError
        If ``zone`` is not a valid value in ``zoneinfo.available_timezones()``.

    Examples
    --------
    >>> from onekit import timekit as tk
    >>> tk.timestamp()  # doctest: +SKIP
    '2024-01-01T00:00:00+00:00'

    >>> tk.timestamp("CET")  # doctest: +SKIP
    '2024-01-01T01:00:00+01:00'

    >>> tk.timestamp("America/New_York")  # doctest: +SKIP
    '2023-12-31T19:00:00-05:00'
    """
    try:
        now = (
            dt.datetime.now().astimezone()
            if timezone is None
            else dt.datetime.now(ZoneInfo(timezone))
        )
        return now.isoformat() if fmt is None else now.strftime(fmt)
    except (IsADirectoryError, ZoneInfoNotFoundError) as error:
        raise InvalidChoiceError(timezone, available_timezones()) from error


def to_datetime(value: DateTimeLike, /, fmt: str | None = None) -> dt.datetime:
    """Return a date, time, or string value as a datetime.datetime object.

    Parameters
    ----------
    value : datetime, date, time, str
        - For datetime: returns input value
        - For date: sets time to 00:00:00
        - For time: sets date to 1900-01-01
        - For string: tries a series of different conversion strategies
    fmt : str, optional
        Provide datetime format to parse from string.
        If None, a series of pre-defined formats are tried.

    Raises
    ------
    TypeError
        If ``value`` is not a datetime, date, time, or string.
    ValueError
        If ``value`` is a string but cannot be converted.

    Examples
    --------
    >>> from onekit import timekit as tk
    >>> tk.to_datetime("2000-01-02T03:04:05.123456")
    datetime.datetime(2000, 1, 2, 3, 4, 5, 123456)

    >>> tk.to_datetime("2020-10-30")
    datetime.datetime(2020, 10, 30, 0, 0)

    >>> tk.to_datetime("00:11:22")
    datetime.datetime(1900, 1, 1, 0, 11, 22)
    """
    if isinstance(value, dt.datetime):
        return value

    def get_fill_dt() -> dt.datetime:
        return dt.datetime(1900, 1, 1, 0, 0, 0, 0)

    match value:
        case str():
            v = value.strip()

            if fmt is not None:
                return dt.datetime.strptime(v, fmt)

            try:
                return dt.datetime.fromisoformat(v)
            except ValueError:
                pass

            fmts = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d_%H:%M:%S",
                "%Y-%m-%d-%H:%M:%S",
                "%Y-%m-%d %H:%M",
                "%Y-%m-%d_%H:%M",
                "%Y-%m-%d-%H:%M",
                "%Y%m%d %H%M%S",
                "%Y%m%d_%H%M%S",
                "%Y%m%d-%H%M%S",
                "%Y%m%d %H%M",
                "%Y%m%d_%H%M",
                "%Y%m%d-%H%M",
                "%d-%m-%Y %H:%M:%S",
                "%d-%m-%Y %H:%M",
                "%Y%m%d",
                "%Y.%m.%d",
                "%Y-%m-%d",
                "%Y/%m/%d",
                "%d.%m.%Y",
                "%d-%m-%Y",
                "%d/%m/%Y",
                "%m/%d/%Y",
                "%H%M",
                "%H:%M",
                "%H%M%S",
                "%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S%Z",
                "%Y-%m-%dT%H:%M:%S%Z%z",
            ]
            fill_dt = get_fill_dt()
            for fmt in fmts:
                try:
                    result = dt.datetime.strptime(v, fmt)
                    if "%Y" not in fmt:
                        result = result.replace(year=fill_dt.year)
                    if "%m" not in fmt:
                        result = result.replace(month=fill_dt.month)
                    if "%d" not in fmt:
                        result = result.replace(day=fill_dt.day)
                    return result

                except ValueError:
                    continue

            try:
                return parser.parse(v, default=fill_dt)
            except (ValueError, OverflowError):
                pass

            raise ValueError(f"{value=!r} - unrecognized datetime format")

        case dt.date():
            return dt.datetime.combine(value, get_fill_dt().time())

        case dt.time():
            return dt.datetime.combine(get_fill_dt().date(), value)

        case _:
            # noinspection PyUnreachableCode
            raise TypeError(f"{type(value)} is unsupported")
