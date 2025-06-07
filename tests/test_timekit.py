import datetime as dt
import platform
import re
import time

import pytest
import pytz
import time_machine
import toolz
from toolz import curried

from onekit import timekit as tk
from onekit.timekit import DateRange


class TestDateRange:

    @pytest.mark.parametrize(
        "min_date, max_date",
        [
            (dt.date(2025, 6, 1), dt.date(2025, 6, 1)),
            (dt.date(2025, 6, 1), dt.date(2025, 6, 7)),
        ],
    )
    def test_create_date_range__valid_call(self, min_date: dt.date, max_date: dt.date):
        actual = tk.create_date_range(min_date, max_date)
        assert isinstance(actual, DateRange)
        assert actual.min_date == min_date
        assert actual.max_date == max_date

    @pytest.mark.parametrize(
        "min_date, max_date",
        [
            (dt.date(2025, 6, 7), dt.date(2025, 6, 1)),
        ],
    )
    def test_create_date_range__reversed(self, min_date: dt.date, max_date: dt.date):
        actual = tk.create_date_range(min_date, max_date)
        assert isinstance(actual, DateRange)
        assert actual.min_date == max_date
        assert actual.max_date == min_date


@pytest.mark.parametrize(
    "n, ref_date, expected",
    [
        (1.0, dt.date(2022, 1, 1), None),
        (-1, dt.date(2022, 1, 1), None),
        (0, dt.date(2022, 1, 1), dt.date(2022, 1, 1)),
        (1, dt.date(2022, 1, 1), dt.date(2021, 12, 31)),
        (2, dt.date(2022, 1, 1), dt.date(2021, 12, 30)),
        (3, dt.date(2022, 1, 1), dt.date(2021, 12, 29)),
        (7, dt.date(2022, 8, 1), dt.date(2022, 7, 25)),
        (30, dt.date(2022, 8, 1), dt.date(2022, 7, 2)),
        (27, dt.date(2022, 2, 1), dt.date(2022, 1, 5)),
        (28, dt.date(2022, 2, 1), dt.date(2022, 1, 4)),
        (27, dt.date(2020, 2, 1), dt.date(2020, 1, 5)),
        (28, dt.date(2020, 2, 1), dt.date(2020, 1, 4)),
        (29, dt.date(2020, 2, 1), dt.date(2020, 1, 3)),
    ],
)
def test_date_ago(n: int, ref_date, expected: dt.date):
    if isinstance(n, int) and n >= 0:
        actual = tk.date_ago(ref_date, n)
        assert actual == expected

        dates = tk.date_range(actual, ref_date, incl_min=True, incl_max=True)
        num_days = curried.count(dates)
        num_days_expected = n + 1
        assert num_days == num_days_expected
    else:
        with pytest.raises(ValueError):
            tk.date_ago(ref_date, n)


@pytest.mark.parametrize(
    "n, ref_date, expected",
    [
        (1.0, dt.date(2022, 1, 1), None),
        (-1, dt.date(2022, 1, 1), None),
        (0, dt.date(2022, 1, 1), dt.date(2022, 1, 1)),
        (1, dt.date(2022, 1, 1), dt.date(2022, 1, 2)),
        (2, dt.date(2022, 1, 1), dt.date(2022, 1, 3)),
        (3, dt.date(2022, 1, 1), dt.date(2022, 1, 4)),
        (7, dt.date(2022, 8, 1), dt.date(2022, 8, 8)),
        (30, dt.date(2022, 8, 1), dt.date(2022, 8, 31)),
        (27, dt.date(2022, 2, 1), dt.date(2022, 2, 28)),
        (28, dt.date(2022, 2, 1), dt.date(2022, 3, 1)),
        (27, dt.date(2020, 2, 1), dt.date(2020, 2, 28)),
        (28, dt.date(2020, 2, 1), dt.date(2020, 2, 29)),
        (29, dt.date(2020, 2, 1), dt.date(2020, 3, 1)),
    ],
)
def test_date_ahead(n: int, ref_date, expected: dt.date):
    if isinstance(n, int) and n >= 0:
        actual = tk.date_ahead(ref_date, n)
        assert actual == expected

        dates = tk.date_range(ref_date, actual, incl_min=True, incl_max=True)
        num_days = curried.count(dates)
        num_days_expected = n + 1
        assert num_days == num_days_expected
    else:
        with pytest.raises(ValueError):
            tk.date_ahead(ref_date, n)


def test_date_count_backward():
    ref_date = dt.date(2022, 1, 1)
    actual = toolz.pipe(tk.date_count_backward(ref_date), curried.take(3), list)
    expected = [dt.date(2022, 1, 1), dt.date(2021, 12, 31), dt.date(2021, 12, 30)]
    assert actual == expected


def test_date_count_forward():
    ref_date = dt.date(2022, 1, 1)
    actual = toolz.pipe(tk.date_count_forward(ref_date), curried.take(3), list)
    expected = [dt.date(2022, 1, 1), dt.date(2022, 1, 2), dt.date(2022, 1, 3)]
    assert actual == expected


def test_date_diff():
    d1 = dt.date(2024, 7, 1)
    d2 = dt.date(2024, 7, 7)

    actual = tk.date_diff(d1, d1)
    expected = 0
    assert actual == expected

    actual = tk.date_diff(d1, d2)
    expected = 6
    assert actual == expected

    actual = tk.date_diff(d2, d1)
    expected = -6
    assert actual == expected


@pytest.mark.parametrize(
    "start, end, expected",
    [
        (
            dt.date(2022, 1, 1),
            dt.date(2022, 1, 5),
            (
                dt.date(2022, 1, 1),
                dt.date(2022, 1, 2),
                dt.date(2022, 1, 3),
                dt.date(2022, 1, 4),
                dt.date(2022, 1, 5),
            ),
        ),
        (
            dt.date(2022, 1, 5),
            dt.date(2022, 1, 1),
            (
                dt.date(2022, 1, 1),
                dt.date(2022, 1, 2),
                dt.date(2022, 1, 3),
                dt.date(2022, 1, 4),
                dt.date(2022, 1, 5),
            ),
        ),
        (dt.date(2022, 1, 1), dt.date(2022, 1, 1), (dt.date(2022, 1, 1),)),
    ],
)
def test_date_range(start: dt.date, end: dt.date, expected: tuple[dt.date]):
    actual = tuple(tk.date_range(start, end))
    assert actual == expected

    actual = tuple(tk.date_range(start, end, incl_min=False))
    assert actual == expected[1:]

    actual = tuple(tk.date_range(start, end, incl_max=False))
    assert actual == expected[:-1]

    actual = tuple(tk.date_range(start, end, incl_min=False, incl_max=False))
    assert actual == expected[1:-1]


@pytest.mark.parametrize(
    "d, expected",
    [
        (dt.date(2022, 1, 1), "2022-01-01"),
        (dt.date(2022, 1, 31), "2022-01-31"),
    ],
)
def test_date_to_str(d: dt.date, expected: str):
    actual = tk.date_to_str(d)
    assert actual == expected


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (-2.0, None),
        (-1, None),
        (0, "0s"),
        (1, "1s"),
        (59, "59s"),
        (59.0, "59s"),
        (60, "1m"),
        (60.1, "1m"),
        (61, "1m 1s"),
        (61.1, "1m 1s"),
        (120, "2m"),
        (120.1, "2m"),
        (60 * 60, "1h"),
        (60 * 60 + 1, "1h 1s"),
        (60 * 60 * 24, "1d"),
        (60 * 60 * 24 + 1, "1d 1s"),
        (110.0, "1m 50s"),
        (0.4142135623730951, "0.414214s"),
        (0.5, "0.5s"),
        (1.4142135623730951, "1.41421s"),
        (1.5, "1.5s"),
        (2.4142135623730951, "2.41421s"),
        (59.4142135623730951, "59.4142s"),
        (60.4142135623730951, "1m"),
        (60.5142135623730951, "1m 1s"),
        (60 * 60 * 24 + 123456, "2d 10h 17m 36s"),
    ],
)
def test_humantime(seconds: int | float, expected: str | None):
    if seconds >= 0:
        actual = tk.humantime(seconds)
        assert actual == expected
    else:
        with pytest.raises(ValueError):
            tk.humantime(seconds)


@pytest.mark.parametrize(
    "d, expected",
    [
        (dt.date(2022, 1, 1), dt.date(2022, 1, 31)),
        (dt.date(2022, 2, 1), dt.date(2022, 2, 28)),
        (dt.date(2022, 3, 1), dt.date(2022, 3, 31)),
        (dt.date(2022, 4, 1), dt.date(2022, 4, 30)),
        (dt.date(2022, 5, 1), dt.date(2022, 5, 31)),
        (dt.date(2022, 6, 1), dt.date(2022, 6, 30)),
        (dt.date(2022, 7, 1), dt.date(2022, 7, 31)),
        (dt.date(2022, 8, 1), dt.date(2022, 8, 31)),
        (dt.date(2022, 9, 1), dt.date(2022, 9, 30)),
        (dt.date(2022, 10, 1), dt.date(2022, 10, 31)),
        (dt.date(2022, 11, 1), dt.date(2022, 11, 30)),
        (dt.date(2022, 12, 1), dt.date(2022, 12, 31)),
        (dt.date(1970, 1, 1), dt.date(1970, 1, 31)),
        (dt.date(1970, 1, 15), dt.date(1970, 1, 31)),
        (dt.date(1970, 1, 31), dt.date(1970, 1, 31)),
        (dt.date(2020, 2, 2), dt.date(2020, 2, 29)),
        (dt.date(2022, 2, 3), dt.date(2022, 2, 28)),
        (dt.date(2000, 2, 4), dt.date(2000, 2, 29)),
        (dt.date(1900, 2, 5), dt.date(1900, 2, 28)),
        (dt.date(2012, 2, 27), dt.date(2012, 2, 29)),
        (dt.date(2012, 2, 28), dt.date(2012, 2, 29)),
        (dt.date(2012, 2, 29), dt.date(2012, 2, 29)),
    ],
)
def test_last_date_of_month(d: dt.date, expected: dt.date):
    actual = tk.last_date_of_month(d.year, d.month)
    assert actual == expected


@pytest.mark.parametrize(
    "d1, d2, expected",
    [
        (dt.date(2022, 8, 1), dt.date(2022, 8, 1), 1),
        (dt.date(2022, 8, 1), dt.date(2022, 8, 7), 7),
        (dt.date(2022, 8, 7), dt.date(2022, 8, 1), 7),
        (dt.date(2014, 1, 1), dt.date(2016, 5, 6), 857),
    ],
)
def test_num_days(d1: dt.date, d2: dt.date, expected: int):
    actual = tk.num_days(d1, d2)
    assert actual == expected


class TestStopwatch:
    def test_context_manager__default_call(
        self,
        slumber,
        regex_default_message,
        capsys,
    ):
        with tk.stopwatch():
            slumber()

        actual = capsys.readouterr().out
        expected = regex_default_message
        assert re.search(expected, actual) is not None

    def test_context_manager__instance(self, slumber, regex_default_message):
        with tk.stopwatch() as sw:
            slumber()

        actual = str(sw)
        expected = regex_default_message
        assert re.search(expected, actual) is not None

    @pytest.mark.parametrize("label", [None, "lbl", 1])
    def test_context_manager__label(
        self,
        slumber,
        regex_default_message,
        label,
    ):
        with tk.stopwatch(label) as sw:
            slumber()

        actual = str(sw)
        expected = (
            regex_default_message
            if label is None
            else regex_default_message.replace("$", f" - {label}$")
        )
        assert re.search(expected, actual) is not None
        assert sw.label is None if label is None else sw.label == label

        with pytest.raises(AttributeError):
            sw.label = label

        with pytest.raises(
            TypeError,
            match=r"got some positional-only arguments passed as keyword arguments",
        ):
            # noinspection PyArgumentList
            with tk.stopwatch(label=label):
                slumber()

    @pytest.mark.parametrize("flush", [True, False])
    def test_context_manager__flush(self, slumber, regex_default_message, flush):
        with tk.stopwatch(flush=flush) as sw:
            slumber()

        actual = str(sw)
        expected = regex_default_message
        assert re.search(expected, actual) is not None
        assert sw.flush == flush

        with pytest.raises(AttributeError):
            sw.flush = flush

    @pytest.mark.parametrize(
        "case,fmt",
        [
            (1, None),
            (2, "%Y-%m-%d %H:%M:%S"),
            (3, "%H:%M:%S"),
            (4, "%A, %d %B %Y %H:%M:%S"),
        ],
    )
    @pytest.mark.parametrize("zone", [None, "UTC", "CET"])
    def test_context_manager__fmt(
        self,
        slumber,
        regex_default_message,
        case,
        zone,
        fmt,
        default_fmt="%Y-%m-%d %H:%M:%S",
    ):
        with tk.stopwatch(timezone=zone, fmt=fmt) as sw:
            slumber()

        actual = str(sw)
        expected = (
            regex_default_message
            if case in (1, 2)
            else (
                self.create_regex_for_message(r"\d{2}:\d{2}:\d{2}")
                if case == 3
                else (
                    self.create_regex_for_message(
                        r"\w+, \d{2} \w+ \d{4} \d{2}:\d{2}:\d{2}"
                    )
                    if case == 4
                    else None
                )
            )
        )
        assert re.search(expected, actual) is not None
        assert sw.fmt == default_fmt if fmt is None else sw.fmt == fmt

        # change timestamp format but not data
        sw.fmt = default_fmt

        with pytest.raises(AttributeError):
            sw.start_time = dt.datetime.now()

        with pytest.raises(AttributeError):
            sw.stop_time = dt.datetime.now()

        with pytest.raises(AttributeError):
            sw.elapsed_time = dt.timedelta(days=42)

        actual = str(sw)
        expected = regex_default_message
        assert re.search(expected, actual) is not None

    @pytest.mark.parametrize("label", [None, "lbl"])
    @pytest.mark.parametrize("flush", [True, False])
    @pytest.mark.parametrize("zone", [None, "UTC", "CET"])
    @pytest.mark.parametrize(
        "case,fmt", [(1, None), (2, "%Y-%m-%d %H:%M:%S"), (3, "%H:%M:%S")]
    )
    def test_context_manager__many_param(
        self,
        slumber,
        regex_default_message,
        label,
        flush,
        zone,
        case,
        fmt,
        default_fmt="%Y-%m-%d %H:%M:%S",
    ):
        with tk.stopwatch(label, flush=flush, timezone=zone, fmt=fmt) as sw:
            slumber()

        actual = str(sw)
        expected_message = (
            regex_default_message
            if case in (1, 2)
            else (
                self.create_regex_for_message(r"\d{2}:\d{2}:\d{2}")
                if case == 3
                else None
            )
        )
        expected = (
            expected_message
            if label is None
            else expected_message.replace("$", f" - {label}$")
        )
        assert re.search(expected, actual) is not None
        assert sw.label == label
        assert sw.flush == flush
        assert sw.fmt == default_fmt if fmt is None else sw.fmt == fmt

        with pytest.raises(AttributeError):
            sw.label = label

        with pytest.raises(AttributeError):
            sw.flush = flush

        with pytest.raises(AttributeError):
            sw.start_time = dt.datetime.now()

        with pytest.raises(AttributeError):
            sw.stop_time = dt.datetime.now()

        with pytest.raises(AttributeError):
            sw.elapsed_time = dt.timedelta(days=42)

    def test_context_manager__total_elapsed_time(self, slumber, regex_default_message):
        with tk.stopwatch(1) as sw1:
            slumber()

        with tk.stopwatch(2) as sw2:
            slumber()

        with tk.stopwatch(3) as sw3:
            slumber()

        for i, sw in enumerate([sw1, sw2, sw3]):
            label = str(i + 1)
            actual = str(sw)
            expected = regex_default_message.replace("$", f" - {label}$")
            assert re.search(expected, actual) is not None

        additions = [
            (1, sum([sw1])),
            (2, sw1 + sw2),
            (3, sum([sw2], start=sw1)),
            (4, sum([sw1, sw2])),
            (5, sw1 + sw2 + sw3),
            (6, sum([sw2, sw3], start=sw1)),
            (7, sum([sw1, sw2, sw3])),
        ]
        for case, total in additions:
            actual = str(total)
            n_stopwatches = 1 if case == 1 else 2 if 2 <= case <= 4 else 3
            expected = rf"^0\.0{n_stopwatches}(\d*)?s - total elapsed time$"
            assert re.search(expected, actual) is not None

    def test_decorator__default_call(self, slumber, regex_default_message, capsys):
        @tk.stopwatch()
        def func():
            slumber()

        func()

        actual = capsys.readouterr().out
        expected = regex_default_message.replace("$", f" - {func.__name__}$")
        assert re.search(expected, actual) is not None

    def test_decorator__label(
        self,
        slumber,
        regex_default_message,
        capsys,
        label="lbl",
    ):
        @tk.stopwatch(label)
        def func():
            slumber()

        func()

        actual = capsys.readouterr().out
        expected = regex_default_message.replace("$", f" - {label}$")
        assert re.search(expected, actual) is not None

    @pytest.mark.parametrize("flush", [True, False])
    def test_decorator__flush(self, slumber, regex_default_message, flush, capsys):
        @tk.stopwatch(flush=flush)
        def func():
            slumber()

        func()

        actual = capsys.readouterr().out
        expected = regex_default_message.replace("$", f" - {func.__name__}$")
        assert re.search(expected, actual) is not None

    @pytest.mark.parametrize(
        "case,fmt",
        [
            (1, None),
            (2, "%Y-%m-%d %H:%M:%S"),
            (3, "%H:%M:%S"),
            (4, "%A, %d %B %Y %H:%M:%S"),
        ],
    )
    @pytest.mark.parametrize("zone", [None, "UTC", "CET"])
    def test_decorator__fmt(
        self,
        slumber,
        regex_default_message,
        zone,
        case,
        fmt,
        capsys,
    ):
        @tk.stopwatch(timezone=zone, fmt=fmt)
        def func():
            slumber()

        func()

        actual = capsys.readouterr().out
        expected_message = (
            regex_default_message
            if case in [1, 2]
            else (
                self.create_regex_for_message(r"\d{2}:\d{2}:\d{2}")
                if case == 3
                else (
                    self.create_regex_for_message(
                        r"\w+, \d{2} \w+ \d{4} \d{2}:\d{2}:\d{2}"
                    )
                    if case == 4
                    else None
                )
            )
        )
        expected = expected_message.replace("$", f" - {func.__name__}$")
        assert re.search(expected, actual) is not None

    @pytest.mark.parametrize("label", [None, "lbl"])
    @pytest.mark.parametrize("flush", [True, False])
    @pytest.mark.parametrize("zone", [None, "UTC", "CET"])
    @pytest.mark.parametrize(
        "case,fmt", [(1, None), (2, "%Y-%m-%d %H:%M:%S"), (3, "%H:%M:%S")]
    )
    def test_decorator__many_param(
        self,
        slumber,
        regex_default_message,
        label,
        flush,
        zone,
        case,
        fmt,
        capsys,
    ):
        @tk.stopwatch(label, timezone=zone, fmt=fmt, flush=flush)
        def func():
            slumber()

        func()

        actual = capsys.readouterr().out
        expected_message = (
            regex_default_message
            if case in (1, 2)
            else (
                self.create_regex_for_message(r"\d{2}:\d{2}:\d{2}")
                if case == 3
                else None
            )
        )
        expected = (
            expected_message.replace("$", f" - {func.__name__}$")
            if label is None
            else expected_message.replace("$", f" - {label}$")
        )
        assert re.search(expected, actual) is not None

    @pytest.mark.parametrize("label", [True, 0.0, set(), [2]])
    def test_raises_type_error__label(self, slumber, label):
        with pytest.raises(
            TypeError,
            match=r"label=.* - must be str, int, or NoneType",
        ):
            with tk.stopwatch(label):
                slumber()

    @pytest.mark.parametrize("flush", [None, 0, 1.0, set(), [2]])
    def test_raises_type_error__flush(self, slumber, flush):
        with pytest.raises(TypeError, match=r"flush=.* - must be bool"):
            # noinspection PyTypeChecker
            with tk.stopwatch(flush=flush):
                slumber()

    @pytest.mark.parametrize("zone", [True, 0, 1.0, set(), [2]])
    def test_raises_type_error__zone(self, slumber, zone):
        with pytest.raises(TypeError, match=r"timezone=.* - must be str or NoneType"):
            # noinspection PyTypeChecker
            with tk.stopwatch(timezone=zone):
                slumber()

    @pytest.mark.parametrize("fmt", [True, 0, 1.0, set(), [2]])
    def test_raises_type_error__fmt(self, slumber, fmt):
        with pytest.raises(TypeError, match=r"fmt=.* - must be str or NoneType"):
            # noinspection PyTypeChecker
            with tk.stopwatch(fmt=fmt):
                slumber()

    @pytest.mark.parametrize("fmt", [True, 0, 1.0, set(), [2]])
    def test_raises_type_error__fmt_setter(self, slumber, fmt):
        with tk.stopwatch() as sw:
            slumber()

        with pytest.raises(TypeError, match=r"value=.* - `fmt` must be str"):
            sw.fmt = fmt

    @pytest.fixture(scope="class")
    def slumber(self):
        def _():
            time.sleep(0.01)

        return _

    @pytest.fixture(scope="class")
    def regex_default_message(self, regex_default_fmt):
        """Regex: default output message."""
        return self.create_regex_for_message(regex_default_fmt)

    @pytest.fixture(scope="class")
    def regex_default_fmt(self):
        """Regex: default timestamp format."""
        return r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"

    @staticmethod
    def create_regex_for_message(regex_fmt):
        return rf"^{regex_fmt} -> {regex_fmt} = 0\.01(\d*)?s$"


@pytest.mark.parametrize(
    "string, expected",
    [
        ("2022-01-01", dt.date(2022, 1, 1)),
        ("2022-01-31", dt.date(2022, 1, 31)),
    ],
)
def test_str_to_date(string: str, expected: dt.date):
    actual = tk.str_to_date(string)
    assert actual == expected


class TestTimestamp:
    def test_default_call(self):
        if platform.system() != "Windows":
            # LD_PRELOAD is only available on Unix platforms
            traveller = time_machine.travel(dt.datetime(2024, 1, 1, 0, 0, 0))
            traveller.start()
            actual = tk.timestamp("UCT")
            expected = "2024-01-01 00:00:00"
            assert actual == expected
            traveller.stop()

    @pytest.mark.parametrize(
        "fmt,expected",
        [
            (None, "2024-01-01 00:00:00"),
            ("%Y-%m-%d %H:%M:%S", "2024-01-01 00:00:00"),
            ("%Y%m%d-%H%M%S", "20240101-000000"),
            ("%H:%M:%S", "00:00:00"),
            ("%A, %d %B %Y %H:%M:%S", "Monday, 01 January 2024 00:00:00"),
        ],
    )
    @pytest.mark.parametrize("zone", ["UTC", "utc"])
    def test_utc(self, zone: str, fmt: str, expected: str):
        traveller = time_machine.travel(dt.datetime(2024, 1, 1, 0, 0, 0))
        traveller.start()
        actual = tk.timestamp(zone, fmt=fmt)
        assert actual == expected
        traveller.stop()

    @pytest.mark.parametrize(
        "fmt,expected",
        [
            (None, "2024-01-01 01:00:00"),
            ("%Y-%m-%d %H:%M:%S", "2024-01-01 01:00:00"),
            ("%Y%m%d-%H%M%S", "20240101-010000"),
            ("%H:%M:%S", "01:00:00"),
            ("%A, %d %B %Y %H:%M:%S", "Monday, 01 January 2024 01:00:00"),
        ],
    )
    @pytest.mark.parametrize("zone", ["CET", "cet"])
    def test_cet(self, zone: str, fmt: str, expected: str):
        traveller = time_machine.travel(dt.datetime(2024, 1, 1, 0, 0, 0))
        traveller.start()
        actual = tk.timestamp(zone, fmt=fmt)
        assert actual == expected
        traveller.stop()

    @pytest.mark.parametrize(
        "fmt,expected",
        [
            (None, "2023-12-31 14:00:00"),
            ("%Y-%m-%d %H:%M:%S", "2023-12-31 14:00:00"),
            ("%Y%m%d-%H%M%S", "20231231-140000"),
            ("%H:%M:%S", "14:00:00"),
            ("%A, %d %B %Y %H:%M:%S", "Sunday, 31 December 2023 14:00:00"),
        ],
    )
    @pytest.mark.parametrize("zone", ["US/Hawaii", "us/hawaii", "Us/HawaiI"])
    def test_hawaii(self, zone: str, fmt: str, expected: str):
        traveller = time_machine.travel(dt.datetime(2024, 1, 1, 0, 0, 0))
        traveller.start()
        actual = tk.timestamp(zone, fmt=fmt)
        assert actual == expected
        traveller.stop()

    @pytest.mark.parametrize(
        "fmt,expected",
        [
            (None, "2024-01-01 09:00:00"),
            ("%Y-%m-%d %H:%M:%S", "2024-01-01 09:00:00"),
            ("%Y%m%d-%H%M%S", "20240101-090000"),
            ("%H:%M:%S", "09:00:00"),
            ("%A, %d %B %Y %H:%M:%S", "Monday, 01 January 2024 09:00:00"),
        ],
    )
    @pytest.mark.parametrize("zone", ["Asia/Tokyo", "asia/tokyo", "AsIa/ToKyO"])
    def test_tokyo(self, zone: str, fmt: str, expected: str):
        traveller = time_machine.travel(dt.datetime(2024, 1, 1, 0, 0, 0))
        traveller.start()
        actual = tk.timestamp(zone, fmt=fmt)
        assert actual == expected
        traveller.stop()

    @pytest.mark.parametrize("zone", ["CTE", "Europe"])
    def test_invalid_input(self, zone: str):
        with pytest.raises(pytz.UnknownTimeZoneError):
            tk.timestamp(zone)


@pytest.mark.parametrize(
    "d, expected",
    [
        (dt.date(2022, 8, 1), "Mon"),
        (dt.date(2022, 8, 2), "Tue"),
        (dt.date(2022, 8, 3), "Wed"),
        (dt.date(2022, 8, 4), "Thu"),
        (dt.date(2022, 8, 5), "Fri"),
        (dt.date(2022, 8, 6), "Sat"),
        (dt.date(2022, 8, 7), "Sun"),
    ],
)
def test_weekday(d: dt.date, expected: str):
    actual = tk.weekday(d)
    assert actual == expected
