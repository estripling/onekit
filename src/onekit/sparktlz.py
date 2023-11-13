"""PySpark toolz."""

import functools

from pyspark.sql import DataFrame as SparkDataFrame

from onekit import pytlz

__all__ = ("union",)


def union(*dataframes: SparkDataFrame) -> SparkDataFrame:
    """Union multiple Spark dataframes by name.

    Examples
    --------
    >>> from pyspark.sql import SparkSession  # doctest: +SKIP
    >>> from onekit import sparktlz  # doctest: +SKIP
    >>> spark = SparkSession.builder.getOrCreate()  # doctest: +SKIP
    >>> df1 = spark.createDataFrame([dict(x=1, y=2), dict(x=3, y=4)])  # doctest: +SKIP
    >>> df2 = spark.createDataFrame([dict(x=5, y=6), dict(x=7, y=8)])  # doctest: +SKIP
    >>> df3 = spark.createDataFrame([dict(x=0, y=1), dict(x=2, y=3)])  # doctest: +SKIP
    >>> sparktlz.union(df1, df2, df3).show()  # doctest: +SKIP
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
    return functools.reduce(SparkDataFrame.unionByName, pytlz.flatten(dataframes))
