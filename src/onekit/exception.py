import os

from pyspark.sql import DataFrame as SparkDF

from onekit import pythonkit as pk


class OnekitError(Exception):
    """A base class for onekit exceptions."""


class ColumnNotFoundError(OnekitError):
    """Exception if columns are not found in dataframe."""

    def __init__(self, missing_cols: list[str]):
        self.missing_cols = missing_cols
        self.message = f"following columns not found: {missing_cols}"
        super().__init__(self.message)


class RowCountMismatchError(OnekitError):
    """Exception if row counts mismatch."""

    def __init__(self, num_lft: int, num_rgt: int):
        num_diff = abs(num_lft - num_rgt)
        self.num_lft = num_lft
        self.num_rgt = num_rgt
        self.num_diff = num_diff
        self.message = f"{num_lft=:_}, {num_rgt=:_}, {num_diff=:_}"
        super().__init__(self.message)


class RowMismatchError(OnekitError):
    """Exception if rows mismatch."""

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
        self.message = f"{num_lft=:_}, {num_rgt=:_}"
        super().__init__(self.message)


class SchemaMismatchError(OnekitError):
    """Exception if schemas mismatch."""

    def __init__(self, lft_schema: str, rgt_schema: str):
        self.lft_schema = lft_schema
        self.rgt_schema = rgt_schema
        msg = pk.highlight_string_differences(lft_schema, rgt_schema)
        num_diff = sum(c == "|" for c in msg.splitlines()[1])
        self.message = pk.concat_strings(os.linesep, f"{num_diff=}", msg)
        super().__init__(self.message)
