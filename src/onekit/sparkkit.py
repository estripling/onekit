import functools
from typing import (
    Callable,
    Iterable,
)

from IPython import get_ipython
from IPython.display import (
    HTML,
    display,
)
from pyspark.sql import DataFrame as SparkDF

import onekit.pythonkit as pk

__all__ = (
    "peek",
    "union",
)

DfIdentityFunction = Callable[[SparkDF], SparkDF]


def peek(
    n: int = 6,
    *,
    shape: bool = False,
    cache: bool = False,
    schema: bool = False,
    index: bool = False,
) -> DfIdentityFunction:
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
    ...     df.transform(sk.peek())
    ...     .where("x IS NOT NULL")
    ...     .transform(sk.peek())
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
        """Evaluate specified `peek` function for given dataframe."""
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
