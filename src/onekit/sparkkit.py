import functools
import os
from typing import (
    Callable,
    Iterable,
    List,
    Union,
)

import toolz
from IPython import get_ipython
from IPython.display import (
    HTML,
    display,
)
from pyspark.sql import Column as SparkCol
from pyspark.sql import DataFrame as SparkDF
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql import types as T
from toolz import curried

import onekit.pythonkit as pk

__all__ = (
    "add_prefix",
    "add_suffix",
    "check_dataframe_equal",
    "check_row_count_equal",
    "check_row_equal",
    "check_schema_equal",
    "count_nulls",
    "cvf",
    "daterange",
    "is_dataframe_equal",
    "is_row_count_equal",
    "is_row_equal",
    "is_schema_equal",
    "join",
    "peek",
    "str_to_col",
    "union",
    "with_endofweek_date",
    "with_index",
    "with_startofweek_date",
    "with_weekday",
)

SparkDFIdentityFunc = Callable[[SparkDF], SparkDF]
SparkDFTransformFunc = Callable[[SparkDF], SparkDF]


class SparkkitError(Exception):
    """A base class for sparkkit exceptions."""


class RowCountMismatchError(SparkkitError):
    """Exception if row counts mismatch."""

    def __init__(self, n_lft: int, n_rgt: int):
        self.n_lft = n_lft
        self.n_rgt = n_rgt
        self.difference = n_lft - n_rgt
        self.ratio = n_lft / n_rgt
        self.message = pk.concat_strings(
            os.linesep,
            f"{n_lft=:,}",
            f"{n_rgt=:,}",
            f"{n_lft - n_rgt=:,}",
            f"{n_lft / n_rgt=:g}",
        )
        super().__init__(self.message)


class RowMismatchError(SparkkitError):
    """Exception if rows mismatch."""

    def __init__(self, lft_rows: SparkDF, rgt_rows: SparkDF, n_lft: int, n_rgt: int):
        self.lft_rows = lft_rows
        self.rgt_rows = rgt_rows
        self.n_lft = n_lft
        self.n_rgt = n_rgt
        self.message = f"{n_lft=:,} {n_rgt=:,}"
        super().__init__(self.message)


class SchemaMismatchError(SparkkitError):
    """Exception if schemas mismatch."""

    def __init__(self, lft_schema: str, rgt_schema: str):
        self.lft_schema = lft_schema
        self.rgt_schema = rgt_schema
        msg = pk.highlight_string_differences(lft_schema, rgt_schema)
        n_differences = sum(c == "|" for c in msg.splitlines()[1])
        self.message = pk.concat_strings(os.linesep, f"{n_differences=}", msg)
        super().__init__(self.message)


def add_prefix(prefix: str, /, *, subset=None) -> SparkDFTransformFunc:
    """Add prefix to column names.

    Examples
    --------
    >>> from pyspark.sql import SparkSession
    >>> import onekit.sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame([dict(x=1, y=2)])
    >>> df.transform(sk.add_prefix("pfx_")).show()
    +-----+-----+
    |pfx_x|pfx_y|
    +-----+-----+
    |    1|    2|
    +-----+-----+
    <BLANKLINE>
    """

    def inner(df: SparkDF, /) -> SparkDF:
        cols = subset or df.columns
        for col in cols:
            df = df.withColumnRenamed(col, f"{prefix}{col}")
        return df

    return inner


def add_suffix(suffix: str, /, *, subset=None) -> SparkDFTransformFunc:
    """Add suffix to column names.

    Examples
    --------
    >>> from pyspark.sql import SparkSession
    >>> import onekit.sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame([dict(x=1, y=2)])
    >>> df.transform(sk.add_suffix("_sfx")).show()
    +-----+-----+
    |x_sfx|y_sfx|
    +-----+-----+
    |    1|    2|
    +-----+-----+
    <BLANKLINE>
    """

    def inner(df: SparkDF, /) -> SparkDF:
        cols = subset or df.columns
        for col in cols:
            df = df.withColumnRenamed(col, f"{col}{suffix}")
        return df

    return inner


def check_dataframe_equal(lft_df: SparkDF, rgt_df: SparkDF, /) -> None:
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
    check_schema_equal : Validate schemas.
    check_row_count_equal : Validate row counts.
    check_row_equal : Validate rows.

    Examples
    --------
    >>> from pyspark.sql import Row, SparkSession
    >>> import onekit.sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> lft_df = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])
    >>> rgt_df = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])
    >>> sk.check_dataframe_equal(lft_df, rgt_df) is None
    True

    >>> lft_df = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])
    >>> rgt_df = spark.createDataFrame([Row(z=1, y="a", x=9), Row(z=3, y="b", x=8)])
    >>> try:
    ...     sk.check_dataframe_equal(lft_df, rgt_df)
    ... except sk.SchemaMismatchError as error:
    ...     print(error)
    ...
    n_differences=15
    struct<x:bigint,y:bigint>
           |          |||  |||||||||||
    struct<z:bigint,y:string,x:bigint>

    >>> lft_df = spark.createDataFrame([Row(x=1, y=2)])
    >>> rgt_df = spark.createDataFrame([Row(x=3, y=4), Row(x=5, y=6)])
    >>> try:
    ...     sk.check_dataframe_equal(lft_df, rgt_df)
    ... except sk.RowCountMismatchError as error:
    ...     print(error)
    ...
    n_lft=1
    n_rgt=2
    n_lft - n_rgt=-1
    n_lft / n_rgt=0.5

    >>> lft_df = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4), Row(x=5, y=6)])
    >>> rgt_df = spark.createDataFrame([Row(x=3, y=4), Row(x=5, y=9), Row(x=7, y=8)])
    >>> try:
    ...     sk.check_dataframe_equal(lft_df, rgt_df)
    ... except sk.RowMismatchError as error:
    ...     print(error)
    ...
    n_lft=2 n_rgt=2
    """
    check_schema_equal(lft_df, rgt_df)
    check_row_count_equal(lft_df, rgt_df)
    check_row_equal(lft_df, rgt_df)


def check_row_count_equal(lft_df: SparkDF, rgt_df: SparkDF, /) -> None:
    """Validate row counts of both dataframes are equal.

    Raises
    ------
    RowCountMismatchError
        If row counts are not equal.

    See Also
    --------
    check_dataframe_equal : Validate dataframes.

    Examples
    --------
    >>> from pyspark.sql import SparkSession
    >>> import onekit.sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> lft_df = spark.createDataFrame([dict(x=1, y=2), dict(x=3, y=4)])
    >>> rgt_df = spark.createDataFrame([dict(x=1, y=2), dict(x=3, y=4)])
    >>> sk.check_row_count_equal(lft_df, rgt_df) is None
    True

    >>> lft_df = spark.createDataFrame([dict(x=1, y=2), dict(x=3, y=4)])
    >>> rgt_df = spark.createDataFrame([dict(x=1)])
    >>> try:
    ...     sk.check_row_count_equal(lft_df, rgt_df)
    ... except sk.RowCountMismatchError as error:
    ...     print(error)
    ...
    n_lft=2
    n_rgt=1
    n_lft - n_rgt=1
    n_lft / n_rgt=2
    """
    n_lft = lft_df.count()
    n_rgt = rgt_df.count()

    if n_lft != n_rgt:
        raise RowCountMismatchError(n_lft, n_rgt)


def check_row_equal(lft_df: SparkDF, rgt_df: SparkDF, /) -> None:
    """Validate rows of both dataframes are equal.

    Raises
    ------
    RowMismatchError
        If rows are not equal.

    See Also
    --------
    check_dataframe_equal : Validate dataframes.

    Examples
    --------
    >>> from pyspark.sql import SparkSession
    >>> import onekit.sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> lft_df = spark.createDataFrame([dict(x=1, y=2), dict(x=3, y=4)])
    >>> rgt_df = spark.createDataFrame([dict(x=1, y=2), dict(x=3, y=4)])
    >>> sk.check_row_equal(lft_df, rgt_df) is None
    True

    >>> lft_df = spark.createDataFrame([dict(x=1, y=2), dict(x=3, y=4)])
    >>> rgt_df = spark.createDataFrame([dict(x=3, y=4), dict(x=5, y=6), dict(x=7, y=8)])
    >>> try:
    ...     sk.check_row_equal(lft_df, rgt_df)
    ... except sk.RowMismatchError as error:
    ...     print(error)
    ...
    n_lft=1 n_rgt=2
    """
    lft_rows = lft_df.subtract(rgt_df)
    rgt_rows = rgt_df.subtract(lft_df)

    n_lft = lft_rows.count()
    n_rgt = rgt_rows.count()

    is_equal = (n_lft == 0) and (n_rgt == 0)

    if not is_equal:
        raise RowMismatchError(lft_rows, rgt_rows, n_lft, n_rgt)


def check_schema_equal(lft_df: SparkDF, rgt_df: SparkDF, /) -> None:
    """Validate schemas of both dataframes are equal.

    Raises
    ------
    sk.SchemaMismatchError
        If schemas are not equal.

    See Also
    --------
    check_dataframe_equal : Validate dataframes.

    Examples
    --------
    >>> from pyspark.sql import SparkSession
    >>> import onekit.sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> lft_df = spark.createDataFrame([dict(x=1, y=2), dict(x=3, y=4)])
    >>> rgt_df = spark.createDataFrame([dict(x=1, y=2), dict(x=3, y=4)])
    >>> sk.check_schema_equal(lft_df, rgt_df) is None
    True

    >>> lft_df = spark.createDataFrame([dict(x=1, y=2), dict(x=3, y=4)])
    >>> rgt_df = spark.createDataFrame([dict(x=1), dict(x=3)])
    >>> try:
    ...     sk.check_schema_equal(lft_df, rgt_df)
    ... except sk.SchemaMismatchError as error:
    ...     print(error)
    ...
    n_differences=10
    struct<x:bigint,y:bigint>
                   ||||||||||
    struct<x:bigint>
    """
    # only check column name and type - ignore nullable property
    lft_schema = lft_df.schema.simpleString()
    rgt_schema = rgt_df.schema.simpleString()

    if lft_schema != rgt_schema:
        raise SchemaMismatchError(lft_schema, rgt_schema)


@toolz.curry
def count_nulls(df: SparkDF, /, *, subset=None) -> SparkDF:
    """Count null values in Spark dataframe.

    Examples
    --------
    >>> from pyspark.sql import SparkSession
    >>> import onekit.sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame(
    ...     [
    ...         dict(x=1, y=2, z=None),
    ...         dict(x=4, y=None, z=6),
    ...         dict(x=10, y=None, z=None),
    ...     ]
    ... )
    >>> sk.count_nulls(df).show()
    +---+---+---+
    |  x|  y|  z|
    +---+---+---+
    |  0|  2|  2|
    +---+---+---+
    <BLANKLINE>

    >>> # function is curried
    >>> df.transform(sk.count_nulls(subset=["x", "z"])).show()
    +---+---+
    |  x|  z|
    +---+---+
    |  0|  2|
    +---+---+
    <BLANKLINE>
    """
    cols = subset or df.columns
    return df.agg(*[F.sum(F.isnull(c).cast(T.LongType())).alias(c) for c in cols])


def cvf(*cols: Iterable[str]) -> SparkDFTransformFunc:
    """Count value frequency.

    Examples
    --------
    >>> from pyspark.sql import SparkSession
    >>> import onekit.sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame(
    ...     [
    ...         dict(x="a"),
    ...         dict(x="c"),
    ...         dict(x="b"),
    ...         dict(x="g"),
    ...         dict(x="h"),
    ...         dict(x="a"),
    ...         dict(x="g"),
    ...         dict(x="a"),
    ...     ]
    ... )
    >>> df.transform(sk.cvf("x")).show()
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

    def inner(df: SparkDF, /) -> SparkDF:
        columns = toolz.pipe(cols, pk.flatten, curried.map(str_to_col), list)
        w0 = Window.partitionBy(F.lit(1))
        w1 = w0.orderBy(F.desc("count"), *columns)

        return (
            df.groupby(columns)
            .count()
            .withColumn("percent", 100 * F.col("count") / F.sum("count").over(w0))
            .withColumn("cumul_count", F.sum("count").over(w1))
            .withColumn("cumul_percent", F.sum("percent").over(w1))
            .orderBy("cumul_count")
        )

    return inner


def daterange(
    df: SparkDF,
    /,
    min_date: str,
    max_date: str,
    id_col: str,
    new_col: str,
) -> SparkDF:
    """Generate sequence of consecutive dates between two dates for each distinct ID.

    Examples
    --------
    >>> from pyspark.sql import SparkSession
    >>> import onekit.sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame(
    ...     [
    ...         dict(id=1),
    ...         dict(id=1),
    ...         dict(id=3),
    ...         dict(id=2),
    ...         dict(id=2),
    ...         dict(id=3),
    ...     ]
    ... )
    >>> (
    ...     sk.daterange(df, "2023-05-01", "2023-05-03", "id", "day")
    ...     .orderBy("id", "day")
    ...     .show()
    ... )
    +---+----------+
    | id|       day|
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
    >>> import onekit.sparkkit as sk
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
        check_schema_equal(lft_df, rgt_df)
        check_row_count_equal(lft_df, rgt_df)
        check_row_equal(lft_df, rgt_df)
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
    >>> import onekit.sparkkit as sk
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
        check_row_count_equal(lft_df, rgt_df)
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
    >>> import onekit.sparkkit as sk
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
        check_row_equal(lft_df, rgt_df)
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
    >>> import onekit.sparkkit as sk
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
        check_schema_equal(lft_df, rgt_df)
        return True
    except SchemaMismatchError:
        return False


def join(
    *dataframes: Iterable[SparkDF],
    on: Union[str, List[str]],
    how: str = "inner",
) -> SparkDF:
    """Join iterable of Spark dataframes.

    Examples
    --------
    >>> from pyspark.sql import SparkSession
    >>> import onekit.sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df1 = spark.createDataFrame([dict(id=1, x="a"), dict(id=2, x="b")])
    >>> df2 = spark.createDataFrame([dict(id=1, y="c"), dict(id=2, y="d")])
    >>> df3 = spark.createDataFrame([dict(id=1, z="e"), dict(id=2, z="f")])
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
    n: int = 6,
    *,
    shape: bool = False,
    cache: bool = False,
    schema: bool = False,
    index: bool = False,
) -> SparkDFIdentityFunc:
    """Peek at dataframe between transformations.

    Examples
    --------
    >>> from pyspark.sql import SparkSession
    >>> import onekit.sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame(
    ...     [
    ...         dict(x=1, y="a"),
    ...         dict(x=3, y=None),
    ...         dict(x=None, y="c"),
    ...     ]
    ... )
    >>> df.show()
    +----+----+
    |   x|   y|
    +----+----+
    |   1|   a|
    |   3|null|
    |null|   c|
    +----+----+
    <BLANKLINE>
    >>> filtered_df = (
    ...     df.transform(sk.peek(shape=True))
    ...     .where("x IS NOT NULL")
    ...     .transform(sk.peek(shape=True))
    ... )
    shape = (3, 2)
       x    y
     1.0    a
     3.0 None
    None    c
    shape = (2, 2)
     x    y
     1    a
     3 None
    """

    def inner(df: SparkDF, /) -> SparkDF:
        df = df if df.is_cached else df.cache() if cache else df

        if schema:
            df.printSchema()

        if shape:
            n_rows = pk.num_to_str(df.count())
            n_cols = pk.num_to_str(len(df.columns))
            print(f"shape = ({n_rows}, {n_cols})")

        if n > 0:
            pandas_df = df.limit(n).toPandas()
            pandas_df.index += 1

            is_inside_notebook = get_ipython() is not None

            df_repr = (
                pandas_df.to_html(index=index, na_rep="None", col_space="20px")
                if is_inside_notebook
                else pandas_df.to_string(index=index, na_rep="None")
            )

            display(HTML(df_repr)) if is_inside_notebook else print(df_repr)

        return df

    return inner


def str_to_col(x: str, /) -> SparkCol:
    """Cast string to Spark column else return ``x``.

    Examples
    --------
    >>> from pyspark.sql import functions as F
    >>> import onekit.sparkkit as sk
    >>> sk.str_to_col("x")
    Column<'x'>

    >>> sk.str_to_col(F.col("x"))
    Column<'x'>
    """
    return F.col(x) if isinstance(x, str) else x


def union(*dataframes: Iterable[SparkDF]) -> SparkDF:
    """Union iterable of Spark dataframes by name.

    Examples
    --------
    >>> from pyspark.sql import SparkSession
    >>> import onekit.sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df1 = spark.createDataFrame([dict(x=1, y=2), dict(x=3, y=4)])
    >>> df2 = spark.createDataFrame([dict(x=5, y=6), dict(x=7, y=8)])
    >>> df3 = spark.createDataFrame([dict(x=0, y=1), dict(x=2, y=3)])
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
    return functools.reduce(SparkDF.unionByName, pk.flatten(dataframes))


def with_endofweek_date(
    date_col: str,
    new_col: str,
    last_weekday: str = "Sun",
) -> SparkDFTransformFunc:
    """Add column with the end of the week date.

    Examples
    --------
    >>> from pyspark.sql import SparkSession
    >>> import onekit.sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame(
    ...     [
    ...         dict(day="2023-05-01"),
    ...         dict(day=None),
    ...         dict(day="2023-05-03"),
    ...         dict(day="2023-05-08"),
    ...         dict(day="2023-05-21"),
    ...     ],
    ... )
    >>> df.transform(sk.with_endofweek_date("day", "endofweek")).show()
    +----------+----------+
    |       day| endofweek|
    +----------+----------+
    |2023-05-01|2023-05-07|
    |      null|      null|
    |2023-05-03|2023-05-07|
    |2023-05-08|2023-05-14|
    |2023-05-21|2023-05-21|
    +----------+----------+
    <BLANKLINE>

    >>> df.transform(sk.with_endofweek_date("day", "endofweek", "Sat")).show()
    +----------+----------+
    |       day| endofweek|
    +----------+----------+
    |2023-05-01|2023-05-06|
    |      null|      null|
    |2023-05-03|2023-05-06|
    |2023-05-08|2023-05-13|
    |2023-05-21|2023-05-27|
    +----------+----------+
    <BLANKLINE>
    """

    def inner(df: SparkDF, /) -> SparkDF:
        tmp_col = "_weekday_"
        return (
            df.transform(with_weekday(date_col, tmp_col))
            .withColumn(
                new_col,
                F.when(F.col(tmp_col).isNull(), None)
                .when(F.col(tmp_col) == last_weekday, F.col(date_col))
                .otherwise(F.next_day(F.col(date_col), last_weekday)),
            )
            .drop(tmp_col)
        )

    return inner


def with_index(new_col: str, /) -> SparkDFTransformFunc:
    """Add column with an index of consecutive positive integers.

    Examples
    --------
    >>> from pyspark.sql import SparkSession
    >>> import onekit.sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame([dict(x="a"), dict(x="b"), dict(x="c"), dict(x="d")])
    >>> df.transform(sk.with_index("idx")).show()
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

    def inner(df: SparkDF, /) -> SparkDF:
        w = Window.partitionBy(F.lit(1)).orderBy(F.monotonically_increasing_id())
        return df.withColumn(new_col, F.row_number().over(w))

    return inner


def with_startofweek_date(
    date_col: str,
    new_col: str,
    last_weekday: str = "Sun",
) -> SparkDFTransformFunc:
    """Add column with the start of the week date.

    Examples
    --------
    >>> from pyspark.sql import SparkSession
    >>> import onekit.sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame(
    ...     [
    ...         dict(day="2023-05-01"),
    ...         dict(day=None),
    ...         dict(day="2023-05-03"),
    ...         dict(day="2023-05-08"),
    ...         dict(day="2023-05-21"),
    ...     ],
    ... )
    >>> df.transform(sk.with_startofweek_date("day", "startofweek")).show()
    +----------+-----------+
    |       day|startofweek|
    +----------+-----------+
    |2023-05-01| 2023-05-01|
    |      null|       null|
    |2023-05-03| 2023-05-01|
    |2023-05-08| 2023-05-08|
    |2023-05-21| 2023-05-15|
    +----------+-----------+
    <BLANKLINE>

    >>> df.transform(sk.with_startofweek_date("day", "startofweek", "Sat")).show()
    +----------+-----------+
    |       day|startofweek|
    +----------+-----------+
    |2023-05-01| 2023-04-30|
    |      null|       null|
    |2023-05-03| 2023-04-30|
    |2023-05-08| 2023-05-07|
    |2023-05-21| 2023-05-21|
    +----------+-----------+
    <BLANKLINE>
    """

    def inner(df: SparkDF, /) -> SparkDF:
        tmp_col = "_endofweek_"
        return (
            df.transform(with_endofweek_date(date_col, tmp_col, last_weekday))
            .withColumn(new_col, F.date_sub(tmp_col, 6))
            .drop(tmp_col)
        )

    return inner


def with_weekday(date_col: str, new_col: str) -> SparkDFTransformFunc:
    """Add column with the name of the weekday.

    Examples
    --------
    >>> from pyspark.sql import SparkSession
    >>> import onekit.sparkkit as sk
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame(
    ...     [dict(day="2023-05-01"), dict(day=None), dict(day="2023-05-03")]
    ... )
    >>> df.transform(sk.with_weekday("day", "weekday")).show()
    +----------+-------+
    |       day|weekday|
    +----------+-------+
    |2023-05-01|    Mon|
    |      null|   null|
    |2023-05-03|    Wed|
    +----------+-------+
    <BLANKLINE>
    """

    def determine_weekday(date_column):
        weekday_int = F.dayofweek(date_column)
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

    def inner(df: SparkDF, /) -> SparkDF:
        return df.withColumn(new_col, determine_weekday(date_col))

    return inner
