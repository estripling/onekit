import functools
from typing import (
    Callable,
    Iterable,
)

import numpy as np
import pandas as pd
import toolz
from pandas import DataFrame as PandasDF
from tabulate import tabulate

from onekit import pythonkit as pk

__all__ = (
    "cvf",
    "display",
    "join",
    "profile",
    "union",
)

PandasDFPipeFunc = Callable[[PandasDF], PandasDF]


def cvf(*cols: str | Iterable[str]) -> PandasDFPipeFunc:
    """Count value frequency.

    Examples
    --------
    >>> import pandas as pd
    >>> from onekit import pandaskit as pdk
    >>> df = pd.DataFrame({"x": ["a", "c", "b", "g", "h", "a", "g", "a"]})
    >>> df.pipe(pdk.cvf("x"))
       x  count  percent  cumul_count  cumul_percent
    0  a      3     37.5            3           37.5
    1  g      2     25.0            5           62.5
    2  b      1     12.5            6           75.0
    3  c      1     12.5            7           87.5
    4  h      1     12.5            8          100.0
    """

    def inner(df: PandasDF, /) -> PandasDF:
        columns = toolz.pipe(cols, pk.flatten, list)
        col_order = [False] + [True] * len(columns)

        counts = df.value_counts(subset=columns, dropna=False, sort=False)
        percents = 100 * counts / counts.sum()

        return (
            join(
                counts.to_frame("count").reset_index(),
                percents.to_frame("percent").reset_index(),
                on=columns,
            )
            .sort_values(["count"] + columns, ascending=col_order)
            .assign(
                cumul_count=lambda df: df["count"].cumsum(),
                cumul_percent=lambda df: df["percent"].cumsum(),
            )
            .reset_index(drop=True)
        )

    return inner


def display(
    df: PandasDF,
    label: str | None = None,
    na_rep: str | None = "NULL",
    drop_original_index: bool = True,
    with_row_numbers: bool = True,
) -> None:  # pragma: no cover
    """Returns a stylized representation of the Pandas dataframe.

    Examples
    --------
    >>> import pandas as pd
    >>> from onekit import pandaskit as pdk
    >>> df = pd.DataFrame([dict(x=1, y=2), dict(x=3, y=4), dict(x=None, y=6)])
    >>> pdk.display(df)
    +----+------+-----+
    |    |    x |   y |
    +====+======+=====+
    |  1 |    1 |   2 |
    +----+------+-----+
    |  2 |    3 |   4 |
    +----+------+-----+
    |  3 | NULL |   6 |
    +----+------+-----+
    """
    with_label = label is not None

    def print_tabulated_df() -> None:
        tabulated_df = tabulate(
            df.replace(np.nan, None),
            headers="keys",
            tablefmt="grid",
            floatfmt="_g",
            intfmt="_g",
            missingval=na_rep,
        )
        if with_label:
            n = max(map(len, tabulated_df.splitlines()))
            print(pk.headline(label, n=n, fillchar="="))
        print(tabulated_df)

    if with_row_numbers:
        df = df.copy().reset_index(drop=drop_original_index)
        df.index += 1

    if pk.get_shell_type() != "notebook":
        print_tabulated_df()
        return

    try:
        from IPython.display import display as notebook_display

        df_style = df.style.set_caption(label) if with_label else df.style
        table_styles = []
        if with_label:
            table_styles.append(
                dict(
                    selector="caption",
                    props=[("font-size", "120%"), ("font-weight", "bold")],
                )
            )

        # Function to apply 'g' formatting
        def general_format(x) -> str:
            if pd.isnull(x):
                return na_rep
            return pk.num_to_str(x)

        df_style = (
            df_style.format(
                na_rep=na_rep,
                formatter={
                    col: general_format
                    for col in df.select_dtypes(include=[np.number]).columns
                },
            )
            .highlight_null(props="color: lightgray; background-color: transparent")
            .set_table_styles(table_styles)
        )

        notebook_display(df_style)

    except (ModuleNotFoundError, ImportError):
        print_tabulated_df()


def join(
    *dataframes: PandasDF,
    on: str | list[str],
    how: str = "inner",
) -> PandasDF:
    """Join iterable of Pandas dataframes with index reset.

    Examples
    --------
    >>> import pandas as pd
    >>> from onekit import pandaskit as pdk
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


def profile(df: PandasDF, /, *, q: list[int] | None = None) -> PandasDF:
    """Profile Pandas dataframe.

    Examples
    --------
    >>> import pandas as pd
    >>> from onekit import pandaskit as pdk
    >>> data = {
    ...     "a": [True, None, False, False, True, False],
    ...     "b": [1] * 6,
    ...     "c": [None] * 6,
    ... }
    >>> pdk.profile(pd.DataFrame(data)).T
                        a          b       c
    type           object      int64  object
    count               5          6       0
    isnull              1          0       6
    isnull_pct  16.666667        0.0   100.0
    unique              2          1       0
    unique_pct  33.333333  16.666667     0.0
    sum               NaN        6.0     NaN
    mean              NaN        1.0     NaN
    std               NaN        0.0     NaN
    skewness          NaN        0.0     NaN
    kurtosis          NaN        0.0     NaN
    min               NaN        1.0     NaN
    q5                NaN        1.0     NaN
    q25               NaN        1.0     NaN
    q50               NaN        1.0     NaN
    q75               NaN        1.0     NaN
    q95               NaN        1.0     NaN
    max               NaN        1.0     NaN
    """
    n_rows, _ = df.shape
    quantiles = q or (5, 25, 50, 75, 95)

    basic_info_df = pd.concat(
        [
            df.dtypes.apply(str).to_frame("type"),
            df.count().to_frame("count"),
            (
                df.isnull()
                .sum()
                .to_frame("isnull")
                .assign(isnull_pct=lambda df: 100 * df["isnull"] / n_rows)
            ),
            (
                df.nunique()
                .to_frame("unique")
                .assign(unique_pct=lambda df: 100 * df["unique"] / n_rows)
            ),
        ],
        axis=1,
    )

    return pd.concat(
        [
            basic_info_df,
            df.sum(numeric_only=True).to_frame("sum"),
            df.mean(numeric_only=True).to_frame("mean"),
            df.std(numeric_only=True, ddof=1).to_frame("std"),
            df.skew(numeric_only=True).to_frame("skewness"),
            df.kurt(numeric_only=True).to_frame("kurtosis"),
            df.min(numeric_only=True).to_frame("min"),
            *[
                df.quantile(q / 100, numeric_only=True).to_frame(f"q{q}")
                for q in quantiles
            ],
            df.max(numeric_only=True).to_frame("max"),
        ],
        axis=1,
    )


def union(*dataframes: PandasDF | Iterable[PandasDF]) -> PandasDF:
    """Union iterable of Pandas dataframes by name with index reset.

    Examples
    --------
    >>> import pandas as pd
    >>> from onekit import pandaskit as pdk
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
