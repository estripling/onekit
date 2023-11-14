"""PySpark toolz."""

import functools
from typing import Sequence

from pyspark.sql import DataFrame as SparkDF

from onekit import pytlz

__all__ = ("union",)


def union(*dataframes: Sequence[SparkDF]) -> SparkDF:
    """Union sequence of Spark dataframes by name.

    Examples
    --------
    >>> from pyspark.sql import SparkSession
    >>> from onekit import sparktlz
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df1 = spark.createDataFrame([dict(x=1, y=2), dict(x=3, y=4)])
    >>> df2 = spark.createDataFrame([dict(x=5, y=6), dict(x=7, y=8)])
    >>> df3 = spark.createDataFrame([dict(x=0, y=1), dict(x=2, y=3)])
    >>> sparktlz.union(df1, df2, df3).show()
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
    return functools.reduce(SparkDF.unionByName, pytlz.flatten(dataframes))
