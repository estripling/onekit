import os
from typing import (
    Any,
    Iterable,
)

from pyspark.sql import DataFrame as SparkDF

from onekit import pythonkit as pk

__all__ = (
    "ColumnNotFoundError",
    "InvalidChoiceError",
    "OnekitError",
    "RowCountMismatchError",
    "RowValueMismatchError",
    "SchemaMismatchError",
)


class OnekitError(Exception):
    """A base class for onekit exceptions."""


class ColumnNotFoundError(OnekitError):
    """Exception for missing columns in dataframe.

    See Also
    --------
    check_column_present : Validate column presence.
    has_column : Evaluate column presence.

    Examples
    --------
    >>> from onekit.exception import ColumnNotFoundError
    >>> error = ColumnNotFoundError(missing_cols=["a", "b", "c"])
    >>> error.message
    "following columns not found: ['a', 'b', 'c']"
    """

    def __init__(self, missing_cols: Iterable[str]):
        self.missing_cols = missing_cols
        self.message = f"following columns not found: {missing_cols}"
        super().__init__(self.message)


class InvalidChoiceError(OnekitError):
    """Exception for invalid choice error.

    Examples
    --------
    >>> from onekit.exception import InvalidChoiceError
    >>> x = 0
    >>> error = InvalidChoiceError(value=x, choices=[1, 2, 3])
    >>> error.message
    'x=0 invalid choice - choose from [1, 2, 3]'
    """

    def __init__(self, value: Any, choices: Iterable[Any] | None = None):
        self.value = value
        self.choices = choices
        msg = f"{pk.parent_varname(value)}={value} invalid choice"
        if choices is not None:
            msg += f" - choose from {choices}"
        self.message = msg
        super().__init__(self.message)


class RowCountMismatchError(OnekitError):
    """Exception for mismatch of row counts.

    See Also
    --------
    assert_row_count_equal : Validate row counts.
    is_row_count_equal : Evaluate row counts.

    Examples
    --------
    >>> from onekit.exception import RowCountMismatchError
    >>> error = RowCountMismatchError(num_lft=10000, num_rgt=12000)
    >>> error.message
    'num_lft=10_000, num_rgt=12_000, num_diff=2_000'
    """

    def __init__(self, num_lft: int, num_rgt: int):
        num_diff = abs(num_lft - num_rgt)
        self.num_lft = num_lft
        self.num_rgt = num_rgt
        self.num_diff = num_diff
        self.message = pk.concat_strings(
            ", ",
            f"num_lft={pk.num_to_str(num_lft)}",
            f"num_rgt={pk.num_to_str(num_rgt)}",
            f"num_diff={pk.num_to_str(num_diff)}",
        )
        super().__init__(self.message)


class RowValueMismatchError(OnekitError):
    """Exception for mismatch of row values.

    See Also
    --------
    assert_row_value_equal : Validate row values.
    is_row_value_equal : Evaluate row values.
    """

    def __init__(
        self,
        lft_rows: SparkDF,
        rgt_rows: SparkDF,
        num_lft: int,
        num_rgt: int,
    ):
        self.lft_rows = lft_rows
        self.rgt_rows = rgt_rows
        self.num_lft = num_lft
        self.num_rgt = num_rgt
        self.message = pk.concat_strings(
            ", ",
            f"num_lft={pk.num_to_str(num_lft)}",
            f"num_rgt={pk.num_to_str(num_rgt)}",
        )
        super().__init__(self.message)


class SchemaMismatchError(OnekitError):
    """Exception for mismatch of schemas.

    See Also
    --------
    assert_schema_equal : Validate schemas.
    is_schema_equal : Evaluate schemas.
    """

    def __init__(self, lft_schema: str, rgt_schema: str):
        self.lft_schema = lft_schema
        self.rgt_schema = rgt_schema
        msg = pk.highlight_string_differences(lft_schema, rgt_schema)
        num_diff = sum(c == "|" for c in msg.splitlines()[1])
        self.message = pk.concat_strings(os.linesep, f"{num_diff=}", msg)
        super().__init__(self.message)
