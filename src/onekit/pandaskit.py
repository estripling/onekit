import functools
from typing import (
    Iterable,
    List,
    Union,
)

import pandas as pd
from pandas import DataFrame as PandasDF

import onekit.pythonkit as pk

__all__ = (
    "join",
    "union",
)


def join(
    *dataframes: Iterable[PandasDF],
    on: Union[str, List[str]],
    how: str = "inner",
) -> PandasDF:
    """Join iterable of Pandas dataframes with index reset.

    Examples
    --------
    >>> import pandas as pd
    >>> import onekit.pandaskit as pdk
    >>> df1 = pd.DataFrame([dict(a=1, b=3), dict(a=2, b=4)])
    >>> df2 = pd.DataFrame([dict(a=1, c=5), dict(a=2, c=6)])
    >>> df3 = pd.DataFrame([dict(a=1, d=7)])
    >>> pdk.join(df1, df2, df3, on="a", how="left")
       a  b  c    d
    0  1  3  5  7.0
    1  2  4  6  NaN
    """
    # re-indexing by default
    return functools.reduce(
        functools.partial(pd.merge, on=on, how=how, suffixes=(False, False), copy=True),
        map(pd.DataFrame, pk.flatten(dataframes)),
    )


def union(*dataframes: Iterable[PandasDF]) -> PandasDF:
    """Union iterable of Pandas dataframes by name with index reset.

    Examples
    --------
    >>> import pandas as pd
    >>> import onekit.pandaskit as pdk
    >>> df1 = pd.DataFrame([dict(x=1, y=2), dict(x=3, y=4)])
    >>> df2 = pd.DataFrame([dict(x=5, y=6), dict(x=7, y=8)])
    >>> df3 = pd.DataFrame([dict(x=0, y=1), dict(x=2, y=3)])
    >>> pdk.union(df1, df2, df3)
       x  y
    0  1  2
    1  3  4
    2  5  6
    3  7  8
    4  0  1
    5  2  3

    >>> df1 = pd.DataFrame([[1, 2], [3, 4]], index=[0, 1])
    >>> df2 = pd.DataFrame([[5, 6], [7, 8]], index=[0, 2])
    >>> pdk.union([df1, df2])
       0  1
    0  1  2
    1  3  4
    2  5  6
    3  7  8

    >>> df1 = pd.DataFrame([[1, 2], [3, 4]], index=[0, 1], columns=["a", "b"])
    >>> df2 = pd.DataFrame([[5, 6], [7, 8]], index=[0, 2], columns=["c", "d"])
    >>> pdk.union([df1, df2])
         a    b    c    d
    0  1.0  2.0  NaN  NaN
    1  3.0  4.0  NaN  NaN
    2  NaN  NaN  5.0  6.0
    3  NaN  NaN  7.0  8.0

    >>> df1 = pd.DataFrame([[1, 2], [3, 4]])
    >>> s1 = pd.Series([5, 6])
    >>> pdk.union(df1, s1)
       0    1
    0  1  2.0
    1  3  4.0
    2  5  NaN
    3  6  NaN

    >>> s1 = pd.Series([1, 2])
    >>> s2 = pd.Series([3, 4])
    >>> s3 = pd.Series([5, 6])
    >>> pdk.union([s1, s2], s3)
       0
    0  1
    1  2
    2  3
    3  4
    4  5
    5  6

    >>> s1 = pd.Series([1, 2], index=[0, 1], name="a")
    >>> s2 = pd.Series([3, 4], index=[1, 2], name="b")
    >>> s3 = pd.Series([5, 6], index=[2, 3], name="c")
    >>> pdk.union(s1, s2, s3)
         a    b    c
    0  1.0  NaN  NaN
    1  2.0  NaN  NaN
    2  NaN  3.0  NaN
    3  NaN  4.0  NaN
    4  NaN  NaN  5.0
    5  NaN  NaN  6.0
    """
    return pd.concat(
        map(pd.DataFrame, pk.flatten(dataframes)),
        axis=0,
        ignore_index=True,
    )
