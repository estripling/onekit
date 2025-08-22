import datetime as dt
import platform
import re
import time
from typing import Callable

import pytest
import time_machine
import toolz

from onekit import pythonkit as pk
from onekit import timekit as tk
from onekit.exception import InvalidChoiceError
from onekit.timekit import DateTimeLike


class TestDuration:

    def test_immutability(self):
        actual = tk.Duration("2024-07-01T13:00:00", "2024-07-01T13:00:01")

        # try modifying any attribute – all should raise AttributeError
        parts = (
            "years",
            "months",
            "days",
            "hours",
            "minutes",
            "seconds",
            "total_seconds",
        )
        for part in parts:
            with pytest.raises(AttributeError):
                setattr(actual, part, 10)

        # try adding a new attribute
        with pytest.raises(AttributeError):
            setattr(actual, "new_attr", 10)

        # try bypassing with __dict__ (should fail due to __slots__)
        assert not hasattr(actual, "__dict__")

    @pytest.mark.parametrize(
        "start, end, iso, default",
        [
            ("13:00:00", "13:00:00", "PT0S", "0s"),
            ("13:00:00", "T13:00:00", "PT0S", "0s"),
            ("T13:00:00", "13:00:00", "PT0S", "0s"),
            ("T13:00:00", "T13:00:00", "PT0S", "0s"),
            ("2024-07-01", "2024-07-01", "PT0S", "0s"),
            ("2024-07-01T13:00:00", "2024-07-01T13:00:00", "PT0S", "0s"),
            ("2024-07-01T13:00:00Z", "2024-07-01T13:00:00Z", "PT0S", "0s"),
            ("2024-07-01T13:00:00+00:00", "2024-07-01T13:00:00+00:00", "PT0S", "0s"),
            ("2024-07-01T13:00:00Z+01:00", "2024-07-01T13:00:00Z+01:00", "PT0S", "0s"),
            ("2024-07-01T13:00:00+01:00", "2024-07-01T13:00:00+01:00", "PT0S", "0s"),
            ("2024-07-01T13:00:00.1", "2024-07-01T13:00:00.1", "PT0S", "0s"),
            ("2024-07-01T13:00:00+01:00", "2024-07-01T14:00:00+02:00", "PT0S", "0s"),
            ("2024-07-01T14:00:00+02:00", "2024-07-01T13:00:00+01:00", "PT0S", "0s"),
        ],
    )
    def test_zero_duration(self, start: str, end: str, iso: str, default: str):
        actual = tk.Duration(start, end)
        assert isinstance(actual, tk.Duration)
        assert actual.is_zero is True
        assert actual.as_iso() == iso
        assert actual.as_default() == default

    @pytest.mark.parametrize(
        "start, end, expected",
        [
            ("2024-07-01T13:00:00+00:00", "2024-07-01T13:00:00+00:00", "0"),
            ("T13:00:00", "T13:00:00", "0"),
            ("T13:00:00", "T13:00:00.123456", "0.123456"),
            ("T13:00:00", "T13:00:00.1234567", "0.123456"),
            ("T13:00:00.999999", "T13:00:00.111111", "0.888888"),
            ("T13:00:00.101010", "T13:00:00.101010", "0"),
            ("T13:00:00.101010", "T13:00:00.010101", "0.090909"),
            ("T13:00:00.1010", "T13:00:00.0101", "0.0909"),
            ("T13:00:00", "T13:00:01", "1"),
            ("T13:00:00", "T13:00:59", "59"),
            ("T13:00:00", "T13:00:59.999999", "59.999999"),
            ("T13:00:00", "T13:00:59.999000", "59.999"),
            ("T13:00:00", "T13:00:59.0", "59"),
            ("T13:00:00", "T13:01:00.0", "0"),
        ],
    )
    def test_formatted_seconds(self, start: str, end: str, expected: str):
        actual = tk.Duration(start, end)
        assert isinstance(actual, tk.Duration)
        assert actual.formatted_seconds == expected

    @pytest.mark.parametrize(
        "start, end, iso, default",
        [
            # 1x date change
            ("2024-07-01", "2025-07-01", "P1Y", "1y"),
            ("2024-07-01", "2024-08-01", "P1M", "1mo"),
            ("2024-07-01", "2024-07-02", "P1D", "1d"),
            ("2024-07-01T13:00:00", "2025-07-01T13:00:00", "P1Y", "1y"),
            ("2024-07-01T13:00:00", "2024-08-01T13:00:00", "P1M", "1mo"),
            ("2024-07-01T13:00:00", "2024-07-02T13:00:00", "P1D", "1d"),
            # 2x date changes
            ("2024-07-01", "2025-08-01", "P1Y1M", "1y 1mo"),
            ("2024-07-01", "2025-07-02", "P1Y1D", "1y 1d"),
            ("2024-07-01", "2024-08-02", "P1M1D", "1mo 1d"),
            ("2024-07-01T13:00:00", "2025-08-01T13:00:00", "P1Y1M", "1y 1mo"),
            ("2024-07-01T13:00:00", "2025-07-02T13:00:00", "P1Y1D", "1y 1d"),
            ("2024-07-01T13:00:00", "2024-08-02T13:00:00", "P1M1D", "1mo 1d"),
            # 3x date changes
            ("2024-07-01", "2025-08-02", "P1Y1M1D", "1y 1mo 1d"),
            ("2024-07-01T13:00:00", "2025-08-02T13:00:00", "P1Y1M1D", "1y 1mo 1d"),
            # 1x time change
            ("13:00:00", "14:00:00", "PT1H", "1h"),
            ("13:00:00", "13:01:00", "PT1M", "1m"),
            ("13:00:00", "13:00:01", "PT1S", "1s"),
            ("2024-07-01T13:00:00", "2024-07-01T14:00:00", "PT1H", "1h"),
            ("2024-07-01T13:00:00", "2024-07-01T13:01:00", "PT1M", "1m"),
            ("2024-07-01T13:00:00", "2024-07-01T13:00:01", "PT1S", "1s"),
            # 2x time changes
            ("13:00:00", "14:01:00", "PT1H1M", "1h 1m"),
            ("13:00:00", "14:00:01", "PT1H1S", "1h 1s"),
            ("13:00:00", "13:01:01", "PT1M1S", "1m 1s"),
            ("2024-07-01T13:00:00", "2024-07-01T14:01:00", "PT1H1M", "1h 1m"),
            ("2024-07-01T13:00:00", "2024-07-01T14:00:01", "PT1H1S", "1h 1s"),
            ("2024-07-01T13:00:00", "2024-07-01T13:01:01", "PT1M1S", "1m 1s"),
            # 3x time changes
            ("13:00:00", "14:01:01", "PT1H1M1S", "1h 1m 1s"),
            ("2024-07-01T13:00:00", "2024-07-01T14:01:01", "PT1H1M1S", "1h 1m 1s"),
            # 1x date time + 3x time changes
            ("2024-07-01T13:00:00", "2025-07-01T14:01:01", "P1YT1H1M1S", "1y 1h 1m 1s"),
            (
                "2024-07-01T13:00:00",
                "2024-08-01T14:01:01",
                "P1MT1H1M1S",
                "1mo 1h 1m 1s",
            ),
            ("2024-07-01T13:00:00", "2024-07-02T14:01:01", "P1DT1H1M1S", "1d 1h 1m 1s"),
            # 2x date times + 3x time changes
            (
                "2024-07-01T13:00:00",
                "2025-08-01T14:01:01",
                "P1Y1MT1H1M1S",
                "1y 1mo 1h 1m 1s",
            ),
            (
                "2024-07-01T13:00:00",
                "2025-07-02T14:01:01",
                "P1Y1DT1H1M1S",
                "1y 1d 1h 1m 1s",
            ),
            (
                "2024-07-01T13:00:00",
                "2024-08-02T14:01:01",
                "P1M1DT1H1M1S",
                "1mo 1d 1h 1m 1s",
            ),
            # 3x date times + 3x time changes
            (
                "2024-07-01T13:00:00",
                "2025-08-02T14:01:01",
                "P1Y1M1DT1H1M1S",
                "1y 1mo 1d 1h 1m 1s",
            ),
            # microseconds
            ("2024-07-01T13:00:00.10Z", "2024-07-01T13:00:00.20Z", "PT0.1S", "0.1s"),
            ("2024-07-01T13:00:00", "2024-07-01T13:00:00.5", "PT0.5S", "0.5s"),
            ("2024-07-01T13:00:00", "2024-07-02T13:00:00.5", "P1DT0.5S", "1d 0.5s"),
            ("2024-07-01T13:00:00", "2024-07-01T13:00:00.123", "PT0.123S", "0.123s"),
            ("2024-07-01T13:00:00", "2024-07-01T13:00:01.123", "PT1.123S", "1.123s"),
            (
                "2024-07-01T13:00:00",
                "2024-07-01T13:01:00.123",
                "PT1M0.123S",
                "1m 0.123s",
            ),
            (
                "2024-07-01T10:11:30.123456+00:00",
                "2024-07-01T10:11:40.246801+00:00",
                "PT10.123345S",
                "10.123345s",
            ),
            #
            ("2024-07-01T13:00:00", "2024-07-02T14:00:00", "P1DT1H", "1d 1h"),
            ("2024-07-01T13:00:00", "2024-07-02T15:15:00", "P1DT2H15M", "1d 2h 15m"),
            (
                "2024-07-01T13:00:00",
                "2024-07-02T15:15:30",
                "P1DT2H15M30S",
                "1d 2h 15m 30s",
            ),
            (
                "2020-04-06T15:00:07Z",
                "2021-07-28T19:18:02+00:00",
                "P1Y3M22DT4H17M55S",
                "1y 3mo 22d 4h 17m 55s",
            ),
            (
                "2021-07-28T19:18:02+01:00",
                "2020-04-06T15:00:07+01:00",
                "P1Y3M22DT4H17M55S",
                "1y 3mo 22d 4h 17m 55s",
            ),
            ("2024-07-01T13:00:00+01:00", "2024-07-01T13:00:00+02:00", "PT1H", "1h"),
            ("2024-07-01T13:00:00+02:00", "2024-07-01T13:00:00+01:00", "PT1H", "1h"),
        ],
    )
    def test_default(self, start: str, end: str, iso: str, default: str):
        actual = tk.Duration(start, end)
        assert isinstance(actual, tk.Duration)
        assert actual.is_zero is False
        assert actual.as_iso() == iso
        assert actual.as_default() == default

    @pytest.mark.parametrize(
        "start, end, iso, compact",
        [
            ("2024-07-01T13:00:00", "2024-07-01T13:00:01", "PT1S", "1s"),
            (
                "2024-07-01T13:00:00",
                "2025-08-02T14:01:01",
                "P1Y1M1DT1H1M1S",
                "397d 1h 1m 1s",
            ),
            (
                "2024-07-01T13:00:00+00:00",
                "2024-07-03T23:17:36+00:00",
                "P2DT10H17M36S",
                "2d 10h 17m 36s",
            ),
            (
                "2024-07-01T13:00:00+00:00",
                "2024-07-04T13:02:00+00:00",
                "P3DT2M",
                "3d 2m",
            ),
            ("2024-07-01", "2025-08-02", "P1Y1M1D", "397d"),
            # microseconds
            ("2024-07-01T13:00:00.10Z", "2024-07-01T13:00:00.20Z", "PT0.1S", "0.1s"),
            ("2024-07-01T13:00:00", "2024-07-01T13:00:00.5", "PT0.5S", "0.5s"),
            ("2024-07-01T13:00:00", "2024-07-02T13:00:00.5", "P1DT0.5S", "1d 0.5s"),
            ("2024-07-01T13:00:00", "2024-07-01T13:00:00.123", "PT0.123S", "0.123s"),
            ("2024-07-01T13:00:00", "2024-07-01T13:00:01.123", "PT1.123S", "1.123s"),
            ("2024-07-01T13:00:00", "2024-07-01T13:01:00.12", "PT1M0.12S", "1m 0.12s"),
        ],
    )
    def test_compact_days(self, start: str, end: str, iso: str, compact: str):
        actual = tk.Duration(start, end)
        assert isinstance(actual, tk.Duration)
        assert actual.is_zero is False
        assert actual.as_iso() == iso
        assert actual.as_compact_days() == compact

    @pytest.mark.parametrize(
        "start, end, iso, compact",
        [
            ("2024-07-01T13:00:00", "2024-07-01T13:00:01", "PT1S", "1s"),
            (
                "2024-07-01T13:00:00",
                "2025-08-02T14:01:01",
                "P1Y1M1DT1H1M1S",
                "1y 1mo 1d 1h 1m 1s",
            ),
            (
                "2024-07-01T13:00:00+00:00",
                "2025-07-09T14:01:01+00:00",
                "P1Y8DT1H1M1S",
                "1y 1w 1d 1h 1m 1s",
            ),
            (
                "2024-07-01T13:00:00+00:00",
                "2025-07-08T14:01:01+00:00",
                "P1Y7DT1H1M1S",
                "1y 1w 1h 1m 1s",
            ),
            (
                "2024-07-01T13:00:00+00:00",
                "2025-07-07T14:01:01+00:00",
                "P1Y6DT1H1M1S",
                "1y 6d 1h 1m 1s",
            ),
            # microseconds
            ("2024-07-01T13:00:00.10Z", "2024-07-01T13:00:00.20Z", "PT0.1S", "0.1s"),
            ("2024-07-01T13:00:00", "2024-07-01T13:00:00.5", "PT0.5S", "0.5s"),
            ("2024-07-01T13:00:00", "2024-07-02T13:00:00.5", "P1DT0.5S", "1d 0.5s"),
            ("2024-07-01T13:00:00", "2024-07-01T13:00:00.123", "PT0.123S", "0.123s"),
            ("2024-07-01T13:00:00", "2024-07-01T13:00:01.123", "PT1.123S", "1.123s"),
            ("2024-07-01T13:00:00", "2024-07-01T13:01:00.12", "PT1M0.12S", "1m 0.12s"),
        ],
    )
    def test_compact_weeks(self, start: str, end: str, iso: str, compact: str):
        actual = tk.Duration(start, end)
        assert isinstance(actual, tk.Duration)
        assert actual.is_zero is False
        assert actual.as_iso() == iso
        assert actual.as_compact_weeks() == compact

    @pytest.mark.parametrize(
        "start, end, iso, total_seconds",
        [
            ("2024-07-01T13:00:00", "2024-07-01T13:00:01", "PT1S", "1s"),
            ("2024-07-01T13:00:00", "2024-07-01T14:01:01", "PT1H1M1S", "3_661s"),
        ],
    )
    def test_total_seconds(self, start: str, end: str, iso: str, total_seconds: str):
        actual = tk.Duration(start, end)
        assert isinstance(actual, tk.Duration)
        assert actual.is_zero is False
        assert actual.as_iso() == iso
        assert actual.as_total_seconds() == total_seconds

    @pytest.mark.parametrize(
        "start, end, iso, custom",
        [
            (
                "2024-07-01T13:00:00",
                "2025-07-01T14:01:00",
                "P1YT1H1M",
                "1 year, 0 months, 0 days, 1 hour, 1 minute, 0 seconds",
            ),
            (
                "2024-07-01T13:00:00",
                "2025-08-02T14:01:01",
                "P1Y1M1DT1H1M1S",
                "1 year, 1 month, 1 day, 1 hour, 1 minute, 1 second",
            ),
        ],
    )
    def test_custom(self, start: str, end: str, iso: str, custom: str):

        def show_all(d: tk.Duration) -> str:
            def multiplier(value: int | float) -> str:
                return "" if value == 1 else "s"

            return pk.concat_strings(
                ", ",
                f"{d.years} year{multiplier(d.years)}",
                f"{d.months} month{multiplier(d.months)}",
                f"{d.days} day{multiplier(d.days)}",
                f"{d.hours} hour{multiplier(d.hours)}",
                f"{d.minutes} minute{multiplier(d.minutes)}",
                f"{d.formatted_seconds} second{multiplier(d.seconds)}",
            )

        actual = tk.Duration(start, end)
        assert isinstance(actual, tk.Duration)
        assert actual.is_zero is False
        assert actual.as_iso() == iso
        assert actual.as_custom(formatter=show_all) == custom


class TestDateSub:

    def test_datesub_official_docs(self):
        date_time_values = [
            (
                "1992-09-30 23:59:59",
                "1992-10-01 01:58:00",
                "hour",
                1,
            ),
            (
                "1992-09-30T23:59:59.012345",
                "1992-10-01T01:58:00.123456",
                "hour",
                1,
            ),
            (
                dt.datetime(1992, 9, 30, 23, 59, 59),
                dt.datetime(1992, 10, 1, 1, 58, 00),
                "hour",
                1,
            ),
            (
                "1992-09-30 23:59:59",
                "1992-10-01 01:58:00",
                "min",
                118,
            ),
            (
                "1992-09-30 23:59:59",
                "1992-10-01 01:58:00",
                "sec",
                7081,
            ),
        ]
        for start, end, part, expected in date_time_values:
            actual = tk.datesub(part, start, end)
            assert actual == expected

        time_values = [
            ("01:02:03", "06:01:03", "hour", 4),
            (dt.time(1, 2, 3), dt.time(6, 1, 3), "hour", 4),
        ]
        for start, end, part, expected in time_values:
            actual = tk.datesub(part, start, end)
            assert actual == expected

    @pytest.mark.parametrize(
        "start, end, expected",
        [
            ("ref_date", "ref_date", 0),
            #
            ("2024-01-01", "ref_date", 0),
            ("ref_date", "2024-01-01", 0),
            #
            ("2023-12-31", "ref_date", 0),
            ("ref_date", "2023-12-31", 0),
            #
            ("ref_date", "2024-12-31", 0),
            ("2024-12-31", "ref_date", 0),
            #
            ("ref_date", "2025-01-01", 0),
            ("2025-01-01", "ref_date", 0),
            #
            ("2023-06-30", "ref_date", 1),
            ("2023-07-01", "ref_date", 1),
            ("2023-07-02", "ref_date", 0),
            #
            ("ref_date", "2023-06-30", -1),
            ("ref_date", "2023-07-01", -1),
            ("ref_date", "2023-07-02", 0),
            #
            ("ref_date", "2025-06-30", 0),
            ("ref_date", "2025-07-01", 1),
            ("ref_date", "2025-07-02", 1),
            #
            ("2025-06-30", "ref_date", 0),
            ("2025-07-01", "ref_date", -1),
            ("2025-07-02", "ref_date", -1),
            #
            ("2022-06-30", "ref_date", 2),
            ("2022-07-01", "ref_date", 2),
            ("2022-07-02", "ref_date", 1),
            #
            ("ref_date", "2026-06-30", 1),
            ("ref_date", "2026-07-01", 2),
            ("ref_date", "2026-07-02", 2),
            #
            ("1970-01-01", "ref_date", 54),
        ],
    )
    def test_datesub__year(self, get_ref_date, start: str, end: str, expected: int):
        for cast_func in self.cast_funcs():
            start_date = toolz.pipe(start, get_ref_date, cast_func)
            end_date = toolz.pipe(end, get_ref_date, cast_func)
            for part in ("year", "years", "y", "yr", "yrs"):
                actual = tk.datesub(part, start_date, end_date)
                assert actual == expected

    @pytest.mark.parametrize(
        "start, end, expected",
        [
            ("ref_date", "ref_date", 0),
            #
            ("2024-06-30", "ref_date", 0),
            ("ref_date", "2024-06-30", 0),
            #
            ("ref_date", "2024-07-02", 0),
            ("2024-07-02", "ref_date", 0),
            #
            ("ref_date", "2024-07-07", 0),
            ("2024-07-07", "ref_date", 0),
            #
            ("2024-01-01", "ref_date", 6),
            ("ref_date", "2024-01-01", -6),
            #
            ("2023-12-31", "ref_date", 6),
            ("ref_date", "2023-12-31", -6),
            #
            ("ref_date", "2024-12-31", 5),
            ("2024-12-31", "ref_date", -5),
            #
            ("ref_date", "2025-01-01", 6),
            ("2025-01-01", "ref_date", -6),
            #
            ("2023-06-30", "ref_date", 12),
            ("2023-07-01", "ref_date", 12),
            ("2023-07-02", "ref_date", 11),
            #
            ("ref_date", "2023-06-30", -12),
            ("ref_date", "2023-07-01", -12),
            ("ref_date", "2023-07-02", -11),
            #
            ("ref_date", "2025-06-30", 11),
            ("ref_date", "2025-07-01", 12),
            ("ref_date", "2025-07-02", 12),
            #
            ("2025-06-30", "ref_date", -11),
            ("2025-07-01", "ref_date", -12),
            ("2025-07-02", "ref_date", -12),
            #
            ("2022-06-30", "ref_date", 24),
            ("2022-07-01", "ref_date", 24),
            ("2022-07-02", "ref_date", 23),
            #
            ("ref_date", "2026-06-30", 23),
            ("ref_date", "2026-07-01", 24),
            ("ref_date", "2026-07-02", 24),
            #
            ("1970-01-01", "ref_date", 654),
        ],
    )
    def test_datesub__month(self, get_ref_date, start: str, end: str, expected: int):
        for cast_func in self.cast_funcs():
            start_date = toolz.pipe(start, get_ref_date, cast_func)
            end_date = toolz.pipe(end, get_ref_date, cast_func)
            for part in ("month", "months", "mon", "mons"):
                actual = tk.datesub(part, start_date, end_date)
                assert actual == expected

    @pytest.mark.parametrize(
        "start, end, expected",
        [
            ("ref_date", "ref_date", 0),
            #
            ("2024-06-30", "ref_date", 1),
            ("ref_date", "2024-06-30", -1),
            #
            ("ref_date", "2024-07-02", 1),
            ("2024-07-02", "ref_date", -1),
            #
            ("ref_date", "2024-07-07", 6),
            ("2024-07-07", "ref_date", -6),
            #
            ("2024-01-01", "ref_date", 182),
            ("ref_date", "2024-01-01", -182),
            #
            ("2023-12-31", "ref_date", 183),
            ("ref_date", "2023-12-31", -183),
            #
            ("ref_date", "2024-12-31", 183),
            ("2024-12-31", "ref_date", -183),
            #
            ("ref_date", "2025-01-01", 184),
            ("2025-01-01", "ref_date", -184),
            #
            ("2023-06-30", "ref_date", 367),
            ("2023-07-01", "ref_date", 366),
            ("2023-07-02", "ref_date", 365),
            #
            ("ref_date", "2023-06-30", -367),
            ("ref_date", "2023-07-01", -366),
            ("ref_date", "2023-07-02", -365),
            #
            ("ref_date", "2025-06-30", 364),
            ("ref_date", "2025-07-01", 365),
            ("ref_date", "2025-07-02", 366),
            #
            ("2025-06-30", "ref_date", -364),
            ("2025-07-01", "ref_date", -365),
            ("2025-07-02", "ref_date", -366),
            #
            ("2022-06-30", "ref_date", 732),
            ("2022-07-01", "ref_date", 731),
            ("2022-07-02", "ref_date", 730),
            #
            ("ref_date", "2026-06-30", 729),
            ("ref_date", "2026-07-01", 730),
            ("ref_date", "2026-07-02", 731),
            #
            ("1970-01-01", "ref_date", 19_905),
        ],
    )
    def test_datesub__day(self, get_ref_date, start: str, end: str, expected: int):
        for cast_func in self.cast_funcs():
            start_date = toolz.pipe(start, get_ref_date, cast_func)
            end_date = toolz.pipe(end, get_ref_date, cast_func)
            for part in ("day", "days", "d", "dayofmonth"):
                actual = tk.datesub(part, start_date, end_date)
                assert actual == expected

    # noinspection PyTypeChecker
    @staticmethod
    def cast_funcs() -> tuple[Callable[[str], dt.date | dt.datetime | str]]:
        return (
            toolz.identity,
            tk.to_datetime,
            lambda x: dt.datetime.combine(tk.to_datetime(x), dt.datetime.min.time()),
        )

    @pytest.fixture(scope="class")
    def get_ref_date(self, request: pytest.FixtureRequest) -> Callable[[str], str]:

        def inner(value: str) -> str:
            return request.getfixturevalue("ref_date") if value == "ref_date" else value

        return inner

    @pytest.fixture(scope="class")
    def ref_date(self) -> str:
        return "2024-07-01"


# noinspection PyUnresolvedReferences
class TestStopwatch:

    def test_context_manager(self, capsys):
        with tk.stopwatch():
            TestStopwatch.slumber()

        actual_result = capsys.readouterr().out
        search_result = re.search(self.get_regex(), actual_result)
        assert search_result is not None

    def test_decorator(self, capsys):
        @tk.stopwatch()
        def func():
            TestStopwatch.slumber()

        func()

        actual_result = capsys.readouterr().out
        search_result = re.search(self.get_regex(func.__name__), actual_result)
        assert search_result is not None

    def test_context_manager__instance(self):
        with tk.stopwatch() as sw:
            TestStopwatch.slumber()

        actual_result = str(sw)
        search_result = re.search(self.get_regex(), actual_result)
        assert search_result is not None

    def test_immutability(self):
        with tk.stopwatch() as sw:
            TestStopwatch.slumber()

        with pytest.raises(AttributeError):
            sw.label = "new_label"

        with pytest.raises(AttributeError):
            sw.timezone = "UCT"

        with pytest.raises(AttributeError):
            sw.fmt = "%A"

        with pytest.raises(AttributeError):
            sw.start = dt.datetime.now()

        with pytest.raises(AttributeError):
            sw.end = dt.datetime.now()

        with pytest.raises(AttributeError):
            sw.duration = dt.timedelta(days=42)

        with pytest.raises(AttributeError):
            sw.final_time = 0

    @pytest.mark.parametrize("label", [None, "lbl", 1])
    def test_context_manager__label(self, label):
        with tk.stopwatch(label) as sw:
            TestStopwatch.slumber()

        actual_result = str(sw)
        search_result = re.search(self.get_regex(label), actual_result)
        assert search_result is not None
        assert sw.label is None if label is None else sw.label == label

        with pytest.raises(AttributeError):
            sw.label = label

    @pytest.mark.parametrize("label", [None, "lbl", 1])
    def test_decorator__label(self, label, capsys):
        @tk.stopwatch(label)
        def func():
            TestStopwatch.slumber()

        func()

        actual_result = capsys.readouterr().out
        label_ = func.__name__ if label is None else label
        search_result = re.search(self.get_regex(label_), actual_result)
        assert search_result is not None

    @pytest.mark.parametrize(
        "case, fmt",
        [
            (1, None),
            (2, "%Y-%m-%d %H:%M:%S"),
            (3, "%H:%M:%S"),
            (4, "%A, %d %B %Y %H:%M:%S"),
        ],
    )
    @pytest.mark.parametrize("timezone", [None, "UTC", "CET"])
    def test_context_manager__fmt(self, case, fmt, timezone):
        with tk.stopwatch(timezone=timezone, fmt=fmt) as sw:
            TestStopwatch.slumber()

        actual_result = str(sw)
        match case:
            case 1:
                pattern = self.get_regex()
            case 2:
                pattern = self.make_regex(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
            case 3:
                pattern = self.make_regex(r"\d{2}:\d{2}:\d{2}")
            case 4:
                pattern = self.make_regex(r"\w+, \d{2} \w+ \d{4} \d{2}:\d{2}:\d{2}")
            case _:
                pattern = None

        search_result = re.search(pattern, actual_result)
        assert search_result is not None
        assert sw.timezone is None if timezone is None else sw.timezone == timezone
        assert sw.fmt is None if fmt is None else sw.fmt == fmt

    @pytest.mark.parametrize(
        "case, fmt",
        [
            (1, None),
            (2, "%Y-%m-%d %H:%M:%S"),
            (3, "%H:%M:%S"),
            (4, "%A, %d %B %Y %H:%M:%S"),
        ],
    )
    @pytest.mark.parametrize("timezone", [None, "UTC", "CET"])
    def test_decorator__fmt(self, case, fmt, timezone, capsys):
        @tk.stopwatch(timezone=timezone, fmt=fmt)
        def func():
            TestStopwatch.slumber()

        func()

        actual_result = capsys.readouterr().out
        match case:
            case 1:
                pattern = self.get_regex()
            case 2:
                pattern = self.make_regex(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
            case 3:
                pattern = self.make_regex(r"\d{2}:\d{2}:\d{2}")
            case 4:
                pattern = self.make_regex(r"\w+, \d{2} \w+ \d{4} \d{2}:\d{2}:\d{2}")
            case _:
                pattern = None

        search_result = re.search(self.get_regex(func.__name__, pattern), actual_result)
        assert search_result is not None

    @pytest.mark.parametrize("label", [None, "lbl"])
    @pytest.mark.parametrize("timezone", [None, "UTC", "CET"])
    @pytest.mark.parametrize(
        "case, fmt", [(1, None), (2, "%Y-%m-%d %H:%M:%S"), (3, "%H:%M:%S")]
    )
    def test_context_manager__many_param(self, label, timezone, case, fmt):
        with tk.stopwatch(label, timezone, fmt) as sw:
            TestStopwatch.slumber()

        actual_result = str(sw)
        match case:
            case 1:
                pattern = self.get_regex()
            case 2:
                pattern = self.make_regex(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
            case 3:
                pattern = self.make_regex(r"\d{2}:\d{2}:\d{2}")
            case 4:
                pattern = self.make_regex(r"\w+, \d{2} \w+ \d{4} \d{2}:\d{2}:\d{2}")
            case _:
                pattern = None

        search_result = re.search(self.get_regex(label, pattern), actual_result)
        assert search_result is not None
        assert sw.fmt is None if fmt is None else sw.fmt == fmt

        with pytest.raises(AttributeError):
            sw.label = "new_label"

        with pytest.raises(AttributeError):
            sw.timezone = "UCT"

        with pytest.raises(AttributeError):
            sw.fmt = "%A"

        with pytest.raises(AttributeError):
            sw.start = dt.datetime.now()

        with pytest.raises(AttributeError):
            sw.end = dt.datetime.now()

        with pytest.raises(AttributeError):
            sw.duration = dt.timedelta(days=42)

        with pytest.raises(AttributeError):
            sw.final_time = 0

    @pytest.mark.parametrize("label", [None, "lbl"])
    @pytest.mark.parametrize("timezone", [None, "UTC", "CET"])
    @pytest.mark.parametrize(
        "case, fmt", [(1, None), (2, "%Y-%m-%d %H:%M:%S"), (3, "%H:%M:%S")]
    )
    def test_decorator__many_param(self, label, timezone, case, fmt, capsys):
        @tk.stopwatch(label, timezone, fmt)
        def func():
            TestStopwatch.slumber()

        func()

        actual_result = capsys.readouterr().out
        match case:
            case 1:
                pattern = self.get_regex()
            case 2:
                pattern = self.make_regex(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
            case 3:
                pattern = self.make_regex(r"\d{2}:\d{2}:\d{2}")
            case 4:
                pattern = self.make_regex(r"\w+, \d{2} \w+ \d{4} \d{2}:\d{2}:\d{2}")
            case _:
                pattern = None

        label_ = func.__name__ if label is None else label
        search_result = re.search(self.get_regex(label_, pattern), actual_result)
        assert search_result is not None

    @pytest.fixture(scope="class", autouse=True)
    def _slumber(self) -> None:
        def _():
            time.sleep(0.01)

        self.__class__.slumber = _

    @staticmethod
    def get_regex(label: str | None = None, pattern: str | None = None) -> str:
        """Regex: default stopwatch final time with optional label."""
        regex = pattern or TestStopwatch.stopwatch_regex
        return regex.rstrip("$") + f" - {label}$" if label else regex

    @pytest.fixture(scope="class", autouse=True)
    def _stopwatch_regex(self, timestamp_regex: str) -> None:
        """Store fixture as class variable and activate it for each test (autouse)."""
        self.__class__.stopwatch_regex = self.make_regex(timestamp_regex)

    @pytest.fixture(scope="class")
    def timestamp_regex(self) -> str:
        """Regex: default timestamp."""
        return r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{0,6})?(\+|\-)\d{2}:\d{2}"

    @staticmethod
    def make_regex(regex_fmt: str) -> str:
        """Regex: default stopwatch final time without label."""
        return rf"^{regex_fmt} -> {regex_fmt} took \d*(\.\d*)?s$"


class TestTimestamp:

    def test_default_call(self, tm_seed: dt.datetime):
        if platform.system().lower() != "windows":
            # LD_PRELOAD is only available on Unix platforms
            with time_machine.travel(tm_seed):
                actual = tk.timestamp("UCT")
                expected = "2024-01-01T00:00:00+00:00"
                assert actual == expected

    @pytest.mark.parametrize(
        "fmt, expected",
        [
            (None, "2024-01-01T00:00:00+00:00"),
            ("%Y-%m-%d %H:%M:%S", "2024-01-01 00:00:00"),
            ("%Y-%m-%d %H:%M:%S %Z", "2024-01-01 00:00:00 UTC"),
            ("%Y-%m-%d %H:%M:%S%z", "2024-01-01 00:00:00+0000"),
            ("%Y-%m-%d %H:%M:%S %Z%z", "2024-01-01 00:00:00 UTC+0000"),
            ("%Y%m%d-%H%M%S", "20240101-000000"),
            ("%H:%M:%S", "00:00:00"),
            ("%A, %d %B %Y %H:%M:%S", "Monday, 01 January 2024 00:00:00"),
        ],
    )
    def test_utc(self, fmt: str, expected: str, tm_seed: dt.datetime):
        with time_machine.travel(tm_seed):
            actual = tk.timestamp("UTC", fmt=fmt)
            assert actual == expected

    @pytest.mark.parametrize(
        "fmt, expected",
        [
            (None, "2024-01-01T01:00:00+01:00"),
            ("%Y-%m-%d %H:%M:%S", "2024-01-01 01:00:00"),
            ("%Y-%m-%d %H:%M:%S %Z", "2024-01-01 01:00:00 CET"),
            ("%Y-%m-%d %H:%M:%S%z", "2024-01-01 01:00:00+0100"),
            ("%Y-%m-%d %H:%M:%S %Z%z", "2024-01-01 01:00:00 CET+0100"),
            ("%Y%m%d-%H%M%S", "20240101-010000"),
            ("%H:%M:%S", "01:00:00"),
            ("%A, %d %B %Y %H:%M:%S", "Monday, 01 January 2024 01:00:00"),
        ],
    )
    def test_cet(self, fmt: str, expected: str, tm_seed: dt.datetime):
        with time_machine.travel(tm_seed):
            actual = tk.timestamp("CET", fmt=fmt)
            assert actual == expected

    @pytest.mark.parametrize(
        "fmt, expected",
        [
            (None, "2023-12-31T14:00:00-10:00"),
            ("%Y-%m-%d %H:%M:%S", "2023-12-31 14:00:00"),
            ("%Y-%m-%d %H:%M:%S %Z", "2023-12-31 14:00:00 HST"),
            ("%Y-%m-%d %H:%M:%S%z", "2023-12-31 14:00:00-1000"),
            ("%Y-%m-%d %H:%M:%S %Z%z", "2023-12-31 14:00:00 HST-1000"),
            ("%Y%m%d-%H%M%S", "20231231-140000"),
            ("%H:%M:%S", "14:00:00"),
            ("%A, %d %B %Y %H:%M:%S", "Sunday, 31 December 2023 14:00:00"),
        ],
    )
    def test_hawaii(self, fmt: str, expected: str, tm_seed: dt.datetime):
        with time_machine.travel(tm_seed):
            actual = tk.timestamp("US/Hawaii", fmt=fmt)
            assert actual == expected

    @pytest.mark.parametrize(
        "fmt, expected",
        [
            (None, "2024-01-01T09:00:00+09:00"),
            ("%Y-%m-%d %H:%M:%S", "2024-01-01 09:00:00"),
            ("%Y-%m-%d %H:%M:%S %Z", "2024-01-01 09:00:00 JST"),
            ("%Y-%m-%d %H:%M:%S%z", "2024-01-01 09:00:00+0900"),
            ("%Y-%m-%d %H:%M:%S %Z%z", "2024-01-01 09:00:00 JST+0900"),
            ("%Y%m%d-%H%M%S", "20240101-090000"),
            ("%H:%M:%S", "09:00:00"),
            ("%A, %d %B %Y %H:%M:%S", "Monday, 01 January 2024 09:00:00"),
        ],
    )
    def test_tokyo(self, fmt: str, expected: str, tm_seed: dt.datetime):
        with time_machine.travel(tm_seed):
            actual = tk.timestamp("Asia/Tokyo", fmt=fmt)
            assert actual == expected

    @pytest.mark.parametrize(
        "timezone, fmt, expected_str, expected_dt",
        [
            (
                "UTC",
                None,
                "2024-01-01T00:00:00+00:00",
                dt.datetime(
                    2024, 1, 1, 0, 0, tzinfo=dt.timezone(dt.timedelta(seconds=0))
                ),
            ),
            (
                "CET",
                None,
                "2024-01-01T01:00:00+01:00",
                dt.datetime(
                    2024, 1, 1, 1, 0, tzinfo=dt.timezone(dt.timedelta(seconds=3600))
                ),
            ),
            (
                "US/Hawaii",
                None,
                "2023-12-31T14:00:00-10:00",
                dt.datetime(
                    2023,
                    12,
                    31,
                    14,
                    0,
                    tzinfo=dt.timezone(dt.timedelta(days=-1, seconds=50400)),
                ),
            ),
            (
                "Asia/Tokyo",
                None,
                "2024-01-01T09:00:00+09:00",
                dt.datetime(
                    2024,
                    1,
                    1,
                    9,
                    0,
                    tzinfo=dt.timezone(dt.timedelta(seconds=32400)),
                ),
            ),
        ],
    )
    def test_timestamp_to_datetime(
        self,
        timezone,
        fmt: str,
        expected_str: str,
        expected_dt: dt.datetime,
        tm_seed: dt.datetime,
    ):
        with time_machine.travel(tm_seed):
            actual = tk.timestamp(timezone, fmt=fmt)
            assert actual == expected_str
            assert tk.to_datetime(actual) == expected_dt

    @pytest.mark.parametrize("timezone", ["CTE", "cte", "Europe"])
    def test_invalid_input(self, timezone):
        with pytest.raises(InvalidChoiceError):
            tk.timestamp(timezone)

    @pytest.fixture(scope="class")
    def tm_seed(self) -> dt.datetime:
        """Return time machine seed."""
        return dt.datetime(2024, 1, 1, 0, 0, 0)


class TestToDatetimeConversion:

    @pytest.mark.parametrize(
        "value, expected",
        [
            # datetime
            (dt.datetime(2024, 7, 1), dt.datetime(2024, 7, 1, 0, 0, 0)),
            ("2024-07-01 00:00", dt.datetime(2024, 7, 1, 0, 0, 0)),
            ("2024-07-01 00:00:00", dt.datetime(2024, 7, 1, 0, 0, 0)),
            ("2024-07-01 12:00:00", dt.datetime(2024, 7, 1, 12, 0, 0)),
            ("20240701_120114", dt.datetime(2024, 7, 1, 12, 1, 14)),
            # date
            (dt.date(2024, 7, 1), dt.datetime(2024, 7, 1)),
            ("2024-07-01", dt.datetime(2024, 7, 1)),
            ("2024-07-01", dt.datetime(2024, 7, 1)),
            ("20240701", dt.datetime(2024, 7, 1)),
            # time
            (dt.time(12, 30), dt.datetime(1900, 1, 1, 12, 30)),
            ("12:30", dt.datetime(1900, 1, 1, 12, 30)),
            ("12:30:00", dt.datetime(1900, 1, 1, 12, 30, 0)),
            ("T12:30:00", dt.datetime(1900, 1, 1, 12, 30, 0)),
            ("00:00", dt.datetime(1900, 1, 1, 0, 0)),
            # ISO 8601
            ("2024-07-01T00:00:00", dt.datetime(2024, 7, 1, 00)),
            ("2024-07-01T12:00:00", dt.datetime(2024, 7, 1, 12)),
            ("2024-07-01T23:59:59", dt.datetime(2024, 7, 1, 23, 59, 59)),
        ],
    )
    def test_to_datetime(self, value: DateTimeLike, expected: str):
        actual = tk.to_datetime(value)
        assert actual.tzinfo is None
        assert actual == expected

    @pytest.mark.parametrize(
        "value, expected",
        [
            (
                "2024-07-01T11:22:33Z",
                dt.datetime(2024, 7, 1, 11, 22, 33, tzinfo=dt.timezone.utc),
            ),
            (
                "2024-07-01T11:22:33+00:00",
                dt.datetime(2024, 7, 1, 11, 22, 33, tzinfo=dt.timezone.utc),
            ),
            (
                "2024-07-01T11:22:33-00:00",
                dt.datetime(2024, 7, 1, 11, 22, 33, tzinfo=dt.timezone.utc),
            ),
            (
                "2024-07-01T11:00:00+01:00",
                dt.datetime(2024, 7, 1, 11, tzinfo=dt.timezone(dt.timedelta(hours=1))),
            ),
            (
                "2024-07-01T11:00:00-01:00",
                dt.datetime(2024, 7, 1, 11, tzinfo=dt.timezone(dt.timedelta(hours=-1))),
            ),
            (
                "2024-07-01T11:00:00+01:00",
                dt.datetime(
                    2024, 7, 1, 11, tzinfo=dt.timezone(dt.timedelta(seconds=3600))
                ),
            ),
        ],
    )
    def test_to_datetime__tzinfo(self, value: DateTimeLike, expected: str):
        actual = tk.to_datetime(value)
        expected_tzname = (
            "UTC"
            if any(value.endswith(tz_suffix) for tz_suffix in ("Z", "+00:00", "-00:00"))
            else f"UTC{value[-6:]}"
        )
        assert actual.tzinfo is not None
        assert actual.tzname() == expected_tzname
        assert actual == expected

    @pytest.mark.parametrize(
        "value_cet, value_dt",
        [
            ("2024-01-01T11:22:33+01:00", dt.datetime(2024, 1, 1, 11, 22, 33)),
            ("2024-07-01T11:22:33+01:00", dt.datetime(2024, 7, 1, 11, 22, 33)),
        ],
    )
    def test_to_datetime__cet_timestamps(self, value_cet: str, value_dt: dt.datetime):
        try:
            actual = tk.to_datetime(value_cet)
            expected = value_dt.replace(tzinfo=dt.timezone(dt.timedelta(seconds=3600)))
            assert actual.tzinfo is not None
            assert actual.tzname() == "UTC+01:00"
            assert actual == expected
        except AssertionError:
            value_cest = value_cet.replace("+01:00", "+02:00")
            actual = tk.to_datetime(value_cest)
            expected = value_dt.replace(tzinfo=dt.timezone(dt.timedelta(seconds=7200)))
            assert actual.tzinfo is not None
            assert actual.tzname() == "UTC+02:00"
            assert actual == expected

    @pytest.mark.parametrize(
        "value, fmt, expected",
        [
            (
                "2024/07/01  12.01.14",
                "%Y/%m/%d %H.%M.%S",
                dt.datetime(2024, 7, 1, 12, 1, 14),
            ),
            (
                "20240701_120114",
                "%Y%m%d_%H%M%S",
                dt.datetime(2024, 7, 1, 12, 1, 14),
            ),
        ],
    )
    def test_to_datetime__fmt(self, value: str, fmt: str, expected: str):
        actual = tk.to_datetime(value, fmt=fmt)
        assert actual == expected

    @pytest.mark.parametrize("value", ["foo", "-"])
    def test_invalid_input_value__value_error(self, value: str):
        with pytest.raises(ValueError):
            tk.to_datetime(value)

    @pytest.mark.parametrize("value", [None, 0, 1.0])
    def test_invalid_input_value__type_error(self, value: str):
        with pytest.raises(TypeError):
            tk.to_datetime(value)
