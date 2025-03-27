import datetime as dt
import functools
import math
import os
from typing import (
    Any,
    Iterable,
)

import toolz
from pyspark.sql import Column as SparkCol
from pyspark.sql import DataFrame as SparkDF
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql import types as SparkColType
from pyspark.sql import types as T
from toolz import curried

from onekit import pandaskit as pdk
from onekit import pythonkit as pk

__all__ = (
    "add_prefix",
    "add_suffix",
    "all_col",
    "any_col",
    "assert_dataframe_equal",
    "assert_row_count_equal",
    "assert_row_equal",
    "assert_schema_equal",
    "bool_to_int",
    "bool_to_str",
    "check_column_present",
    "count_nulls",
    "cvf",
    "date_range",
    "filter_date",
    "has_column",
    "is_dataframe_equal",
    "is_row_count_equal",
    "is_row_equal",
    "is_schema_equal",
    "join",
    "peek",
    "select_col_types",
    "str_to_col",
    "union",
    "with_date_diff_ago",
    "with_date_diff_ahead",
    "with_digitscale",
    "with_endofweek_date",
    "with_increasing_id",
    "with_index",
    "with_startofweek_date",
    "with_weekday",
)


class SparkkitError(Exception):
    """A base class for sparkkit exceptions."""


class ColumnNotFoundError(SparkkitError):
    """Exception if columns are not found in dataframe."""

    def __init__(self, missing_cols: list[str]):
        self.missing_cols = missing_cols
        self.message = f"following columns not found: {missing_cols}"
        super().__init__(self.message)


class RowCountMismatchError(SparkkitError):
    """Exception if row counts mismatch."""

    def __init__(self, n_lft: int, n_rgt: int):
        n_diff = abs(n_lft - n_rgt)
        self.n_lft = n_lft
        self.n_rgt = n_rgt
        self.n_diff = n_diff
        self.message = f"{n_lft=:_}, {n_rgt=:_}, {n_diff=:_}"
        super().__init__(self.message)


class RowMismatchError(SparkkitError):
    """Exception if rows mismatch."""

    def __init__(self, lft_rows: SparkDF, rgt_rows: SparkDF, n_lft: int, n_rgt: int):
        self.lft_rows = lft_rows
        self.rgt_rows = rgt_rows
        self.n_lft = n_lft
        self.n_rgt = n_rgt
        self.message = f"{n_lft=:_}, {n_rgt=:_}"
        super().__init__(self.message)


class SchemaMismatchError(SparkkitError):
    """Exception if schemas mismatch."""

    def __init__(self, lft_schema: str, rgt_schema: str):
        self.lft_schema = lft_schema
        self.rgt_schema = rgt_schema
        msg = pk.highlight_string_differences(lft_schema, rgt_schema)
        n_diff = sum(c == "|" for c in msg.splitlines()[1])
        self.message = pk.concat_strings(os.linesep, f"{n_diff=}", msg)
        super().__init__(self.message)


def add_prefix(df: SparkDF, prefix: str, subset: list[str] | None = None) -> SparkDF:
    """Add prefix to column names.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from onekit import sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame([Row(x=1, y=2)])
    >>> sk.add_prefix(df, "pfx_").show()
    +-----+-----+
    |pfx_x|pfx_y|
    +-----+-----+
    |    1|    2|
    +-----+-----+
    <BLANKLINE>
    """
    cols = subset or df.columns
    for col in cols:
        df = df.withColumnRenamed(col, f"{prefix}{col}")
    return df


def add_suffix(df: SparkDF, suffix: str, subset: list[str] | None = None) -> SparkDF:
    """Add suffix to column names.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from onekit import sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame([Row(x=1, y=2)])
    >>> sk.add_suffix(df, "_sfx").show()
    +-----+-----+
    |x_sfx|y_sfx|
    +-----+-----+
    |    1|    2|
    +-----+-----+
    <BLANKLINE>
    """
    cols = subset or df.columns
    for col in cols:
        df = df.withColumnRenamed(col, f"{col}{suffix}")
    return df


def all_col(*cols: str | Iterable[str]) -> SparkCol:
    """Evaluate if all columns are true.

    Notes
    -----
    A NULL value is considered false.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from onekit import sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame(
    ...     [
    ...         Row(x=True, y=True),
    ...         Row(x=True, y=False),
    ...         Row(x=True, y=None),
    ...         Row(x=None, y=False),
    ...         Row(x=None, y=None),
    ...     ]
    ... )
    >>> df.withColumn("all_cols_true", sk.all_col("x", "y")).show()
    +----+-----+-------------+
    |   x|    y|all_cols_true|
    +----+-----+-------------+
    |true| true|         true|
    |true|false|        false|
    |true| NULL|        false|
    |NULL|false|        false|
    |NULL| NULL|        false|
    +----+-----+-------------+
    <BLANKLINE>
    """
    return functools.reduce(
        SparkCol.__and__,
        toolz.pipe(
            cols,
            pk.flatten,
            curried.map(lambda col: F.coalesce(str_to_col(col), F.lit(False))),
        ),
    )


def any_col(*cols: str | Iterable[str]) -> SparkCol:
    """Evaluate if any column is true.

    Notes
    -----
    A NULL value is considered false.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from onekit import sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame(
    ...     [
    ...         Row(x=True, y=True),
    ...         Row(x=True, y=False),
    ...         Row(x=True, y=None),
    ...         Row(x=None, y=False),
    ...         Row(x=None, y=None),
    ...     ]
    ... )
    >>> df.withColumn("any_col_true", sk.any_col("x", "y")).show()
    +----+-----+------------+
    |   x|    y|any_col_true|
    +----+-----+------------+
    |true| true|        true|
    |true|false|        true|
    |true| NULL|        true|
    |NULL|false|       false|
    |NULL| NULL|       false|
    +----+-----+------------+
    <BLANKLINE>
    """
    return functools.reduce(
        SparkCol.__or__,
        toolz.pipe(
            cols,
            pk.flatten,
            curried.map(lambda col: F.coalesce(str_to_col(col), F.lit(False))),
        ),
    )


def assert_dataframe_equal(lft_df: SparkDF, rgt_df: SparkDF, /) -> None:
    """Validate both dataframes are equal.

    Raises
    ------
    SchemaMismatchError
        If schemas are not equal.
    RowCountMismatchError
        If row counts are not equal.
    RowMismatchError
        If rows are not equal.

    See Also
    --------
    assert_schema_equal : Validate schemas.
    assert_row_count_equal : Validate row counts.
    assert_row_equal : Validate rows.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from onekit import sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> lft_df = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])
    >>> rgt_df = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])
    >>> sk.assert_dataframe_equal(lft_df, rgt_df) is None
    True

    >>> lft_df = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])
    >>> rgt_df = spark.createDataFrame([Row(z=1, y="a", x=9), Row(z=3, y="b", x=8)])
    >>> try:
    ...     sk.assert_dataframe_equal(lft_df, rgt_df)
    ... except sk.SchemaMismatchError as error:
    ...     print(error)
    ...
    n_diff=15
    struct<x:bigint,y:bigint>
           |          |||  |||||||||||
    struct<z:bigint,y:string,x:bigint>

    >>> lft_df = spark.createDataFrame([Row(x=1, y=2)])
    >>> rgt_df = spark.createDataFrame([Row(x=3, y=4), Row(x=5, y=6)])
    >>> try:
    ...     sk.assert_dataframe_equal(lft_df, rgt_df)
    ... except sk.RowCountMismatchError as error:
    ...     print(error)
    ...
    n_lft=1, n_rgt=2, n_diff=1

    >>> lft_df = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4), Row(x=5, y=6)])
    >>> rgt_df = spark.createDataFrame([Row(x=3, y=4), Row(x=5, y=9), Row(x=7, y=8)])
    >>> try:
    ...     sk.assert_dataframe_equal(lft_df, rgt_df)
    ... except sk.RowMismatchError as error:
    ...     print(error)
    ...
    n_lft=2, n_rgt=2
    """
    assert_schema_equal(lft_df, rgt_df)
    assert_row_count_equal(lft_df, rgt_df)
    assert_row_equal(lft_df, rgt_df)


def assert_row_count_equal(lft_df: SparkDF, rgt_df: SparkDF, /) -> None:
    """Validate row counts of both dataframes are equal.

    Raises
    ------
    RowCountMismatchError
        If row counts are not equal.

    See Also
    --------
    assert_dataframe_equal : Validate dataframes.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from onekit import sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> lft_df = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])
    >>> rgt_df = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])
    >>> sk.assert_row_count_equal(lft_df, rgt_df) is None
    True

    >>> lft_df = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])
    >>> rgt_df = spark.createDataFrame([Row(x=1)])
    >>> try:
    ...     sk.assert_row_count_equal(lft_df, rgt_df)
    ... except sk.RowCountMismatchError as error:
    ...     print(error)
    ...
    n_lft=2, n_rgt=1, n_diff=1
    """
    n_lft = lft_df.count()
    n_rgt = rgt_df.count()

    if n_lft != n_rgt:
        raise RowCountMismatchError(n_lft, n_rgt)


def assert_row_equal(lft_df: SparkDF, rgt_df: SparkDF, /) -> None:
    """Validate rows of both dataframes are equal.

    Raises
    ------
    RowMismatchError
        If rows are not equal.

    See Also
    --------
    assert_dataframe_equal : Validate dataframes.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from onekit import sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> lft_df = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])
    >>> rgt_df = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])
    >>> sk.assert_row_equal(lft_df, rgt_df) is None
    True

    >>> lft_df = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])
    >>> rgt_df = spark.createDataFrame([Row(x=3, y=4), Row(x=5, y=6), Row(x=7, y=8)])
    >>> try:
    ...     sk.assert_row_equal(lft_df, rgt_df)
    ... except sk.RowMismatchError as error:
    ...     print(error)
    ...
    n_lft=1, n_rgt=2
    """
    lft_rows = lft_df.subtract(rgt_df)
    rgt_rows = rgt_df.subtract(lft_df)

    n_lft = lft_rows.count()
    n_rgt = rgt_rows.count()

    is_equal = (n_lft == 0) and (n_rgt == 0)

    if not is_equal:
        raise RowMismatchError(lft_rows, rgt_rows, n_lft, n_rgt)


def assert_schema_equal(lft_df: SparkDF, rgt_df: SparkDF, /) -> None:
    """Validate schemas of both dataframes are equal.

    Raises
    ------
    sk.SchemaMismatchError
        If schemas are not equal.

    See Also
    --------
    assert_dataframe_equal : Validate dataframes.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from onekit import sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> lft_df = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])
    >>> rgt_df = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])
    >>> sk.assert_schema_equal(lft_df, rgt_df) is None
    True

    >>> lft_df = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])
    >>> rgt_df = spark.createDataFrame([Row(x=1), Row(x=3)])
    >>> try:
    ...     sk.assert_schema_equal(lft_df, rgt_df)
    ... except sk.SchemaMismatchError as error:
    ...     print(error)
    ...
    n_diff=10
    struct<x:bigint,y:bigint>
                   ||||||||||
    struct<x:bigint>
    """
    # only check column name and type - ignore nullable property
    lft_schema = lft_df.schema.simpleString()
    rgt_schema = rgt_df.schema.simpleString()

    if lft_schema != rgt_schema:
        raise SchemaMismatchError(lft_schema, rgt_schema)


def bool_to_int(df: SparkDF, subset: list[str] | None = None) -> SparkDF:
    """Cast values of Boolean columns to 0/1 integer values.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from onekit import sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame(
    ...     [
    ...         Row(x=True, y=False, z=None),
    ...         Row(x=False, y=None, z=True),
    ...         Row(x=True, y=None, z=None),
    ...     ]
    ... )
    >>> sk.bool_to_int(df).show()
    +---+----+----+
    |  x|   y|   z|
    +---+----+----+
    |  1|   0|NULL|
    |  0|NULL|   1|
    |  1|NULL|NULL|
    +---+----+----+
    <BLANKLINE>

    >>> sk.bool_to_int(df, subset=["y", "z"]).show()
    +-----+----+----+
    |    x|   y|   z|
    +-----+----+----+
    | true|   0|NULL|
    |false|NULL|   1|
    | true|NULL|NULL|
    +-----+----+----+
    <BLANKLINE>
    """
    cols = subset or df.columns
    bool_cols = [c for c in select_col_types(df, T.BooleanType) if c in cols]
    for bool_col in bool_cols:
        df = df.withColumn(bool_col, F.col(bool_col).cast(T.IntegerType()))
    return df


def bool_to_str(df: SparkDF, subset: list[str] | None = None) -> SparkDF:
    """Cast values of Boolean columns to string values.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from onekit import sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame(
    ...     [
    ...         Row(x=True, y=False, z=None),
    ...         Row(x=False, y=None, z=True),
    ...         Row(x=True, y=None, z=None),
    ...     ]
    ... )
    >>> sk.bool_to_str(df).show()
    +-----+-----+----+
    |    x|    y|   z|
    +-----+-----+----+
    | true|false|NULL|
    |false| NULL|true|
    | true| NULL|NULL|
    +-----+-----+----+
    <BLANKLINE>

    >>> sk.bool_to_str(df, subset=["y", "z"]).printSchema()
    root
     |-- x: boolean (nullable = true)
     |-- y: string (nullable = true)
     |-- z: string (nullable = true)
    <BLANKLINE>
    """
    cols = subset or df.columns
    bool_cols = [c for c in select_col_types(df, T.BooleanType) if c in cols]
    for bool_col in bool_cols:
        df = df.withColumn(bool_col, F.col(bool_col).cast(T.StringType()))
    return df


def check_column_present(df: SparkDF, *cols: str | Iterable[str]) -> SparkDF:
    """Check if columns are present in dataframe.

    Raises
    ------
    sk.ColumnNotFoundError
        If columns are not found in dataframe.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from onekit import sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame([Row(x=1), Row(x=2), Row(x=3)])
    >>> sk.check_column_present(df, "x").show()
    +---+
    |  x|
    +---+
    |  1|
    |  2|
    |  3|
    +---+
    <BLANKLINE>

    >>> try:
    ...     sk.check_column_present(df, "y").show()
    ... except sk.ColumnNotFoundError as error:
    ...     print(error)
    ...
    following columns not found: ['y']
    """
    missing_cols = [col for col in pk.flatten(cols) if col not in df.columns]
    if len(missing_cols) > 0:
        raise ColumnNotFoundError(missing_cols)
    return df


def count_nulls(df: SparkDF, subset: list[str] | None = None) -> SparkDF:
    """Count NULL values in Spark dataframe.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from onekit import sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame(
    ...     [
    ...         Row(x=1, y=2, z=None),
    ...         Row(x=4, y=None, z=6),
    ...         Row(x=10, y=None, z=None),
    ...     ]
    ... )
    >>> sk.count_nulls(df).show()
    +---+---+---+
    |  x|  y|  z|
    +---+---+---+
    |  0|  2|  2|
    +---+---+---+
    <BLANKLINE>
    """
    cols = subset or df.columns
    return df.agg(*[F.sum(F.isnull(c).cast(T.LongType())).alias(c) for c in cols])


def cvf(df: SparkDF, *cols: str | SparkCol | Iterable[str | SparkCol]) -> SparkDF:
    """Count value frequency.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from onekit import sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame(
    ...     [
    ...         Row(x="a"),
    ...         Row(x="c"),
    ...         Row(x="b"),
    ...         Row(x="g"),
    ...         Row(x="h"),
    ...         Row(x="a"),
    ...         Row(x="g"),
    ...         Row(x="a"),
    ...     ]
    ... )
    >>> sk.cvf(df, "x").show()
    +---+-----+-------+-----------+-------------+
    |  x|count|percent|cumul_count|cumul_percent|
    +---+-----+-------+-----------+-------------+
    |  a|    3|   37.5|          3|         37.5|
    |  g|    2|   25.0|          5|         62.5|
    |  b|    1|   12.5|          6|         75.0|
    |  c|    1|   12.5|          7|         87.5|
    |  h|    1|   12.5|          8|        100.0|
    +---+-----+-------+-----------+-------------+
    <BLANKLINE>
    """
    columns = toolz.pipe(cols, pk.flatten, curried.map(str_to_col), list)
    w0 = Window.partitionBy(F.lit(True))
    w1 = w0.orderBy(F.desc("count"), *columns)
    return (
        df.groupby(columns)
        .count()
        .withColumn("percent", 100 * F.col("count") / F.sum("count").over(w0))
        .withColumn("cumul_count", F.sum("count").over(w1))
        .withColumn("cumul_percent", F.sum("percent").over(w1))
        .orderBy("cumul_count")
    )


def date_range(
    df: SparkDF,
    min_date: str | dt.date,
    max_date: str | dt.date,
    id_col: str,
    new_col: str,
) -> SparkDF:
    """Generate sequence of consecutive dates between two dates for each distinct ID.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from onekit import sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame(
    ...     [
    ...         Row(id=1),
    ...         Row(id=1),
    ...         Row(id=3),
    ...         Row(id=2),
    ...         Row(id=2),
    ...         Row(id=3),
    ...     ]
    ... )
    >>> (
    ...     sk.date_range(df, "2023-05-01", "2023-05-03", "id", "d")
    ...     .orderBy("id", "d")
    ...     .show()
    ... )
    +---+----------+
    | id|         d|
    +---+----------+
    |  1|2023-05-01|
    |  1|2023-05-02|
    |  1|2023-05-03|
    |  2|2023-05-01|
    |  2|2023-05-02|
    |  2|2023-05-03|
    |  3|2023-05-01|
    |  3|2023-05-02|
    |  3|2023-05-03|
    +---+----------+
    <BLANKLINE>
    """
    return (
        df.select(id_col)
        .distinct()
        .withColumn("min_date", F.to_date(F.lit(min_date), "yyyy-MM-dd"))
        .withColumn("max_date", F.to_date(F.lit(max_date), "yyyy-MM-dd"))
        .select(
            id_col,
            F.expr("sequence(min_date, max_date, interval 1 day)").alias(new_col),
        )
        .withColumn(new_col, F.explode(new_col))
    )


def filter_date(
    df: SparkDF,
    date_col: str,
    ref_date: str | dt.date,
    num_days: int | float,
) -> SparkDF:
    """Filter dataframe such that `num_days` corresponds to the number of distinct dates
    when consecutively counting backwards from reference date.

    Notes
    -----
    - reference date is inclusive
    - relative date = date_ago(ref_date, num_days) is exclusive
    - If `num_days=float("inf")`, returns all dates prior to reference date (inclusive)

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from onekit import sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame(
    ...     [
    ...         Row(d="2024-01-01"),
    ...         Row(d="2024-01-02"),
    ...         Row(d="2024-01-03"),
    ...         Row(d="2024-01-04"),
    ...         Row(d="2024-01-05"),
    ...         Row(d="2024-01-06"),
    ...         Row(d="2024-01-07"),
    ...         Row(d="2024-01-08"),
    ...     ],
    ... )
    >>> sk.filter_date(df, "d", ref_date="2024-01-07", num_days=3).show()
    +----------+
    |         d|
    +----------+
    |2024-01-05|
    |2024-01-06|
    |2024-01-07|
    +----------+
    <BLANKLINE>

    >>> sk.filter_date(df, "d", ref_date="2024-01-07", num_days=float("inf")).show()
    +----------+
    |         d|
    +----------+
    |2024-01-01|
    |2024-01-02|
    |2024-01-03|
    |2024-01-04|
    |2024-01-05|
    |2024-01-06|
    |2024-01-07|
    +----------+
    <BLANKLINE>
    """
    if not isinstance(num_days, (int, float)):
        raise TypeError(f"{type(num_days)=} - must be an int or float")

    if isinstance(num_days, int) and num_days < 1:
        raise ValueError(f"{num_days=} - must be a positive integer")

    if isinstance(num_days, float) and not math.isinf(num_days):
        raise ValueError(f'{num_days=} - only valid float value: float("inf")')

    date_diff_ago = "_date_diff_ago_"

    return (
        with_date_diff_ago(df, date_col, ref_date, new_col=date_diff_ago)
        .where((F.col(date_diff_ago) >= 0) & (F.col(date_diff_ago) < num_days))
        .drop(date_diff_ago)
    )


def has_column(df: SparkDF, *cols: str | Iterable[str]) -> bool:
    """Evaluate if all columns are present in dataframe.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from onekit import sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame([Row(x=1), Row(x=2), Row(x=3)])
    >>> sk.has_column(df, "x")
    True

    >>> sk.has_column(df, ["y"])
    False
    """
    try:
        check_column_present(df, cols)
        return True
    except ColumnNotFoundError:
        return False


def is_dataframe_equal(lft_df: SparkDF, rgt_df: SparkDF, /) -> bool:
    """Evaluate if both dataframes are equal.

    See Also
    --------
    is_schema_equal : Evaluate schemas.
    is_row_count_equal : Evaluate row counts.
    is_row_equal : Evaluate rows.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from onekit import sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> lft_df = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])
    >>> rgt_df = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])
    >>> sk.is_dataframe_equal(lft_df, rgt_df)
    True

    >>> lft_df = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])
    >>> rgt_df = spark.createDataFrame([Row(z=1, y="a", x=9), Row(z=3, y="b", x=8)])
    >>> sk.is_dataframe_equal(lft_df, rgt_df)
    False

    >>> lft_df = spark.createDataFrame([Row(x=1, y=2)])
    >>> rgt_df = spark.createDataFrame([Row(x=3, y=4), Row(x=5, y=6)])
    >>> sk.is_dataframe_equal(lft_df, rgt_df)
    False

    >>> lft_df = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4), Row(x=5, y=6)])
    >>> rgt_df = spark.createDataFrame([Row(x=3, y=4), Row(x=5, y=9), Row(x=7, y=8)])
    >>> sk.is_dataframe_equal(lft_df, rgt_df)
    False
    """
    try:
        assert_schema_equal(lft_df, rgt_df)
        assert_row_count_equal(lft_df, rgt_df)
        assert_row_equal(lft_df, rgt_df)
        return True
    except SparkkitError:
        return False


def is_row_count_equal(lft_df: SparkDF, rgt_df: SparkDF, /) -> bool:
    """Evaluate if row counts of both dataframes are equal.

    See Also
    --------
    is_dataframe_equal : Evaluate dataframes.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from onekit import sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> lft_df = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])
    >>> rgt_df = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])
    >>> sk.is_row_count_equal(lft_df, rgt_df)
    True

    >>> lft_df = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])
    >>> rgt_df = spark.createDataFrame([Row(x=1)])
    >>> sk.is_row_count_equal(lft_df, rgt_df)
    False
    """
    try:
        assert_row_count_equal(lft_df, rgt_df)
        return True
    except RowCountMismatchError:
        return False


def is_row_equal(lft_df: SparkDF, rgt_df: SparkDF, /) -> bool:
    """Evaluate if rows of both dataframes are equal.

    See Also
    --------
    is_dataframe_equal : Evaluate dataframes.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from onekit import sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> lft_df = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])
    >>> rgt_df = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])
    >>> sk.is_row_equal(lft_df, rgt_df)
    True

    >>> lft_df = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])
    >>> rgt_df = spark.createDataFrame([Row(x=3, y=4), Row(x=5, y=6), Row(x=7, y=8)])
    >>> sk.is_row_equal(lft_df, rgt_df)
    False
    """
    try:
        assert_row_equal(lft_df, rgt_df)
        return True
    except RowMismatchError:
        return False


def is_schema_equal(lft_df: SparkDF, rgt_df: SparkDF, /) -> bool:
    """Evaluate if schemas of both dataframes are equal.

    See Also
    --------
    is_dataframe_equal : Evaluate dataframes.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from onekit import sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> lft_df = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])
    >>> rgt_df = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])
    >>> sk.is_schema_equal(lft_df, rgt_df)
    True

    >>> lft_df = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])
    >>> rgt_df = spark.createDataFrame([Row(x=1), Row(x=3)])
    >>> sk.is_schema_equal(lft_df, rgt_df)
    False
    """
    try:
        assert_schema_equal(lft_df, rgt_df)
        return True
    except SchemaMismatchError:
        return False


def join(
    *dataframes: SparkDF | Iterable[SparkDF],
    on: str | list[str],
    how: str = "inner",
) -> SparkDF:
    """Join iterable of Spark dataframes.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from onekit import sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df1 = spark.createDataFrame([Row(id=1, x="a"), Row(id=2, x="b")])
    >>> df2 = spark.createDataFrame([Row(id=1, y="c"), Row(id=2, y="d")])
    >>> df3 = spark.createDataFrame([Row(id=1, z="e"), Row(id=2, z="f")])
    >>> sk.join(df1, df2, df3, on="id").show()
    +---+---+---+---+
    | id|  x|  y|  z|
    +---+---+---+---+
    |  1|  a|  c|  e|
    |  2|  b|  d|  f|
    +---+---+---+---+
    <BLANKLINE>
    """
    return functools.reduce(
        functools.partial(SparkDF.join, on=on, how=how),
        pk.flatten(dataframes),
    )


def peek(
    df: SparkDF,
    n: int = 6,
    shape: bool = False,
    cache: bool = False,
    schema: bool = False,
    label: str | None = None,
) -> SparkDF:
    """Peek at dataframe between transformations.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from onekit import sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame(
    ...     [
    ...         Row(x=1, y="a"),
    ...         Row(x=3, y=None),
    ...         Row(x=None, y="c"),
    ...     ]
    ... )
    >>> df.show()
    +----+----+
    |   x|   y|
    +----+----+
    |   1|   a|
    |   3|NULL|
    |NULL|   c|
    +----+----+
    <BLANKLINE>
    >>> filtered_df = (
    ...     sk.peek(df, shape=True)
    ...     .where("x IS NOT NULL")
    ...     .transform(lambda df: sk.peek(df, shape=True))
    ... )
    shape=(3, 2)
    +----+------+------+
    |    |    x | y    |
    +====+======+======+
    |  1 |    1 | a    |
    +----+------+------+
    |  2 |    3 | NULL |
    +----+------+------+
    |  3 | NULL | c    |
    +----+------+------+
    shape=(2, 2)
    +----+-----+------+
    |    |   x | y    |
    +====+=====+======+
    |  1 |   1 | a    |
    +----+-----+------+
    |  2 |   3 | NULL |
    +----+-----+------+
    """
    df = df if df.is_cached else df.cache() if cache else df

    if schema:
        df.printSchema()

    if shape:
        num_rows = pk.num_to_str(df.count())
        num_cols = pk.num_to_str(len(df.columns))
        print(f"shape=({num_rows}, {num_cols})")

    if n > 0:
        pdk.display(
            df=df.limit(n).transform(lambda df: bool_to_str(df)).toPandas(),
            label=label,
        )

    return df


def select_col_types(
    df: SparkDF,
    *col_types: SparkColType | Iterable[SparkColType],
) -> list[str]:
    """Identify columns of specified data type.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from pyspark.sql import types as T
    >>> from onekit import sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame(
    ...     [Row(bool=True, double=1.0, float=2.0, int=3, long=4, str="string")],
    ...     schema=T.StructType(
    ...         [
    ...             T.StructField("bool", T.BooleanType(), nullable=True),
    ...             T.StructField("double", T.DoubleType(), nullable=True),
    ...             T.StructField("float", T.FloatType(), nullable=True),
    ...             T.StructField("int", T.IntegerType(), nullable=True),
    ...             T.StructField("long", T.LongType(), nullable=True),
    ...             T.StructField("str", T.StringType(), nullable=True),
    ...         ]
    ...     ),
    ... )
    >>> sk.select_col_types(df, T.BooleanType)
    ['bool']

    >>> sk.select_col_types(df, T.IntegerType, T.LongType)
    ['int', 'long']
    """

    def get_valid_column_data_types() -> set[str]:
        # noinspection PyTypeChecker
        all_column_data_types: dict[str, Any] = SparkColType.__dict__
        valid_column_data_types = set()
        for k, v in all_column_data_types.items():
            try:
                if k.endswith("Type") and hasattr(v, "typeName"):
                    valid_column_data_types.add(v.typeName())
            except AttributeError:  # pragma: no cover
                continue  # pragma: no cover
        return valid_column_data_types

    col_types = tuple(pk.flatten(col_types))
    valid_types = get_valid_column_data_types()

    for col_type in col_types:
        if not hasattr(col_type, "typeName") or col_type.typeName() not in valid_types:
            raise TypeError(f"{col_type=} - must be a valid data type: {valid_types}")

    return [c for c in df.columns if isinstance(df.schema[c].dataType, col_types)]


def str_to_col(x: str | SparkCol, /) -> SparkCol:
    """Cast string ``x`` to Spark column else return ``x``.

    Examples
    --------
    >>> from pyspark.sql import functions as F
    >>> from onekit import sparkkit as sk
    >>> sk.str_to_col("x")
    Column<'x'>

    >>> sk.str_to_col(F.col("x"))
    Column<'x'>
    """
    return F.col(x) if isinstance(x, str) else x


def union(*dataframes: SparkDF | Iterable[SparkDF]) -> SparkDF:
    """Union iterable of Spark dataframes by name.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from onekit import sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df1 = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])
    >>> df2 = spark.createDataFrame([Row(x=5, y=6), Row(x=7, y=8)])
    >>> df3 = spark.createDataFrame([Row(x=0, y=1), Row(x=2, y=3)])
    >>> sk.union(df1, df2, df3).show()
    +---+---+
    |  x|  y|
    +---+---+
    |  1|  2|
    |  3|  4|
    |  5|  6|
    |  7|  8|
    |  0|  1|
    |  2|  3|
    +---+---+
    <BLANKLINE>
    """
    # noinspection PyTypeChecker
    return functools.reduce(SparkDF.unionByName, pk.flatten(dataframes))


def with_date_diff_ago(
    df: SparkDF,
    date_col: str,
    ref_date: str | dt.date,
    new_col: str,
) -> SparkDF:
    """Add a column showing date differences with respect to the reference date,
    where differences to past dates are positive integers.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from onekit import sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame(
    ...     [
    ...         Row(d="2024-01-01"),
    ...         Row(d="2024-01-02"),
    ...         Row(d="2024-01-03"),
    ...         Row(d="2024-01-04"),
    ...         Row(d="2024-01-05"),
    ...         Row(d="2024-01-06"),
    ...         Row(d="2024-01-07"),
    ...         Row(d="2024-01-08"),
    ...         Row(d="2024-01-09"),
    ...     ],
    ... )
    >>> sk.with_date_diff_ago(df, "d", "2024-01-07", "diff").show()
    +----------+----+
    |         d|diff|
    +----------+----+
    |2024-01-01|   6|
    |2024-01-02|   5|
    |2024-01-03|   4|
    |2024-01-04|   3|
    |2024-01-05|   2|
    |2024-01-06|   1|
    |2024-01-07|   0|
    |2024-01-08|  -1|
    |2024-01-09|  -2|
    +----------+----+
    <BLANKLINE>
    """
    return df.withColumn(new_col, F.datediff(F.lit(ref_date), str_to_col(date_col)))


def with_date_diff_ahead(
    df: SparkDF,
    date_col: str,
    ref_date: str | dt.date,
    new_col: str,
) -> SparkDF:
    """Add a column showing date differences with respect to the reference date,
    where differences to future dates are positive integers.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from onekit import sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame(
    ...     [
    ...         Row(d="2024-01-01"),
    ...         Row(d="2024-01-02"),
    ...         Row(d="2024-01-03"),
    ...         Row(d="2024-01-04"),
    ...         Row(d="2024-01-05"),
    ...         Row(d="2024-01-06"),
    ...         Row(d="2024-01-07"),
    ...         Row(d="2024-01-08"),
    ...         Row(d="2024-01-09"),
    ...     ],
    ... )
    >>> sk.with_date_diff_ahead(df, "d", "2024-01-07", "diff").show()
    +----------+----+
    |         d|diff|
    +----------+----+
    |2024-01-01|  -6|
    |2024-01-02|  -5|
    |2024-01-03|  -4|
    |2024-01-04|  -3|
    |2024-01-05|  -2|
    |2024-01-06|  -1|
    |2024-01-07|   0|
    |2024-01-08|   1|
    |2024-01-09|   2|
    +----------+----+
    <BLANKLINE>
    """
    return df.withColumn(new_col, F.datediff(str_to_col(date_col), F.lit(ref_date)))


# noinspection PyTypeChecker
def with_digitscale(
    df: SparkDF,
    num_col: str,
    new_col: str,
    kind: str = "log",
) -> SparkDF:
    """PySpark version of digitscale.

    See Also
    --------
    onekit.mathkit.digitscale : Python version
    onekit.numpykit.digitscale : NumPy version

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from onekit import sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame(
    ...     [
    ...         Row(x=0.1),
    ...         Row(x=1.0),
    ...         Row(x=10.0),
    ...         Row(x=100.0),
    ...         Row(x=1_000.0),
    ...         Row(x=10_000.0),
    ...         Row(x=100_000.0),
    ...         Row(x=1_000_000.0),
    ...         Row(x=2_000_000.0),
    ...         Row(x=None),
    ...     ],
    ... )
    >>> sk.with_digitscale(df, "x", "fx").show()
    +---------+-----------------+
    |        x|               fx|
    +---------+-----------------+
    |      0.1|              0.0|
    |      1.0|              1.0|
    |     10.0|              2.0|
    |    100.0|              3.0|
    |   1000.0|              4.0|
    |  10000.0|              5.0|
    | 100000.0|              6.0|
    |1000000.0|              7.0|
    |2000000.0|7.301029995663981|
    |     NULL|             NULL|
    +---------+-----------------+
    <BLANKLINE>

    >>> sk.with_digitscale(df, "x", "fx", kind="int").show()
    +---------+----+
    |        x|  fx|
    +---------+----+
    |      0.1|   0|
    |      1.0|   1|
    |     10.0|   2|
    |    100.0|   3|
    |   1000.0|   4|
    |  10000.0|   5|
    | 100000.0|   6|
    |1000000.0|   7|
    |2000000.0|   7|
    |     NULL|NULL|
    +---------+----+
    <BLANKLINE>

    >>> sk.with_digitscale(df, "x", "fx", kind="linear").show()
    +---------+-----------------+
    |        x|               fx|
    +---------+-----------------+
    |      0.1|              0.0|
    |      1.0|              1.0|
    |     10.0|              2.0|
    |    100.0|              3.0|
    |   1000.0|              4.0|
    |  10000.0|              5.0|
    | 100000.0|              6.0|
    |1000000.0|              7.0|
    |2000000.0|7.111111111111111|
    |     NULL|             NULL|
    +---------+-----------------+
    <BLANKLINE>
    """
    valid_kind = ["log", "int", "linear"]
    if kind not in valid_kind:
        raise ValueError(f"{kind=} - must be a valid value: {valid_kind}")

    x = F.abs(num_col)
    df = df.withColumn(
        new_col,
        F.when(x.isNull(), None).when(x >= 0.1, 1 + F.log10(x)).otherwise(0.0),
    )

    if kind == "int":
        df = df.withColumn(new_col, F.floor(new_col).cast(T.IntegerType()))

    if kind == "linear":
        n = "_n_"
        y0 = F.col(n)
        y1 = F.col(n) + 1
        x0 = 10 ** (F.col(n) - 1)
        x1 = 10 ** F.col(n)

        df = (
            df.withColumn(n, F.floor(new_col).cast(T.IntegerType()))
            .withColumn(
                new_col,
                F.when(x.isNull(), None)
                .when(x >= 0.1, (y0 * (x1 - x) + y1 * (x - x0)) / (x1 - x0))
                .otherwise(0.0),
            )
            .drop(n)
        )

    return df


def with_endofweek_date(
    df: SparkDF,
    date_col: str,
    new_col: str,
    last_weekday: str = "Sun",
) -> SparkDF:
    """Add column with the end of the week date.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from onekit import sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame(
    ...     [
    ...         Row(d="2023-05-01"),
    ...         Row(d=None),
    ...         Row(d="2023-05-03"),
    ...         Row(d="2023-05-08"),
    ...         Row(d="2023-05-21"),
    ...     ],
    ... )
    >>> sk.with_endofweek_date(df, "d", "endofweek").show()
    +----------+----------+
    |         d| endofweek|
    +----------+----------+
    |2023-05-01|2023-05-07|
    |      NULL|      NULL|
    |2023-05-03|2023-05-07|
    |2023-05-08|2023-05-14|
    |2023-05-21|2023-05-21|
    +----------+----------+
    <BLANKLINE>

    >>> sk.with_endofweek_date(df, "d", "endofweek", "Sat").show()
    +----------+----------+
    |         d| endofweek|
    +----------+----------+
    |2023-05-01|2023-05-06|
    |      NULL|      NULL|
    |2023-05-03|2023-05-06|
    |2023-05-08|2023-05-13|
    |2023-05-21|2023-05-27|
    +----------+----------+
    <BLANKLINE>
    """
    tmp_col = "_weekday_"
    return (
        with_weekday(df, date_col, tmp_col)
        .withColumn(
            new_col,
            F.when(F.col(tmp_col).isNull(), None)
            .when(F.col(tmp_col) == last_weekday, F.col(date_col))
            .otherwise(F.next_day(F.col(date_col), last_weekday)),
        )
        .drop(tmp_col)
    )


def with_increasing_id(df: SparkDF, new_col: str, /) -> SparkDF:
    """Add column with monotonically increasing id.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from onekit import sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame([Row(x="a"), Row(x="b"), Row(x="c"), Row(x="d")])
    >>> sk.with_increasing_id(df, "id").show()  # doctest: +SKIP
    +---+-----------+
    |  x|         id|
    +---+-----------+
    |  a| 8589934591|
    |  b|25769803776|
    |  c|42949672960|
    |  d|60129542144|
    +---+-----------+
    <BLANKLINE>
    """
    return df.withColumn(new_col, F.monotonically_increasing_id())  # pragma: no cover


def with_index(df: SparkDF, new_col: str) -> SparkDF:
    """Add column with an index of consecutive positive integers.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from onekit import sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame([Row(x="a"), Row(x="b"), Row(x="c"), Row(x="d")])
    >>> sk.with_index(df, "idx").show()
    +---+---+
    |  x|idx|
    +---+---+
    |  a|  1|
    |  b|  2|
    |  c|  3|
    |  d|  4|
    +---+---+
    <BLANKLINE>
    """
    w = Window.partitionBy(F.lit(1)).orderBy(F.monotonically_increasing_id())
    return df.withColumn(new_col, F.row_number().over(w))


def with_startofweek_date(
    df: SparkDF,
    date_col: str,
    new_col: str,
    last_weekday: str = "Sun",
) -> SparkDF:
    """Add column with the start of the week date.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from onekit import sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame(
    ...     [
    ...         Row(d="2023-05-01"),
    ...         Row(d=None),
    ...         Row(d="2023-05-03"),
    ...         Row(d="2023-05-08"),
    ...         Row(d="2023-05-21"),
    ...     ],
    ... )
    >>> sk.with_startofweek_date(df, "d", "startofweek").show()
    +----------+-----------+
    |         d|startofweek|
    +----------+-----------+
    |2023-05-01| 2023-05-01|
    |      NULL|       NULL|
    |2023-05-03| 2023-05-01|
    |2023-05-08| 2023-05-08|
    |2023-05-21| 2023-05-15|
    +----------+-----------+
    <BLANKLINE>

    >>> sk.with_startofweek_date(df, "d", "startofweek", "Sat").show()
    +----------+-----------+
    |         d|startofweek|
    +----------+-----------+
    |2023-05-01| 2023-04-30|
    |      NULL|       NULL|
    |2023-05-03| 2023-04-30|
    |2023-05-08| 2023-05-07|
    |2023-05-21| 2023-05-21|
    +----------+-----------+
    <BLANKLINE>
    """
    tmp_col = "_endofweek_"
    return (
        with_endofweek_date(df, date_col, tmp_col, last_weekday)
        .withColumn(new_col, F.date_sub(tmp_col, 6))
        .drop(tmp_col)
    )


def with_weekday(df: SparkDF, date_col: str, new_col: str) -> SparkDF:
    """Add column with the name of the weekday.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> from onekit import sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame(
    ...     [Row(d="2023-05-01"), Row(d=None), Row(d="2023-05-03")]
    ... )
    >>> sk.with_weekday(df, "d", "weekday").show()
    +----------+-------+
    |         d|weekday|
    +----------+-------+
    |2023-05-01|    Mon|
    |      NULL|   NULL|
    |2023-05-03|    Wed|
    +----------+-------+
    <BLANKLINE>
    """

    def determine_weekday(date_col: str, /) -> SparkCol:
        weekday_int = F.dayofweek(date_col)
        return (
            F.when(weekday_int == 1, "Sun")
            .when(weekday_int == 2, "Mon")
            .when(weekday_int == 3, "Tue")
            .when(weekday_int == 4, "Wed")
            .when(weekday_int == 5, "Thu")
            .when(weekday_int == 6, "Fri")
            .when(weekday_int == 7, "Sat")
            .otherwise(None)
        )

    return df.withColumn(new_col, determine_weekday(date_col))
