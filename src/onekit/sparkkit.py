import functools
from typing import (
    Callable,
    Iterable,
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
from toolz import curried

import onekit.pythonkit as pk

__all__ = (
    "cvf",
    "peek",
    "str_to_col",
    "union",
)

SparkDFIdentityFunc = Callable[[SparkDF], SparkDF]
SparkDFTransformFunc = Callable[[SparkDF], SparkDF]


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
            num_rows = pk.num_to_str(df.count())
            num_cols = pk.num_to_str(len(df.columns))
            print(f"shape = ({num_rows}, {num_cols})")

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
