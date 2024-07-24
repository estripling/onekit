from typing import (
    Optional,
    Union,
)

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas import DataFrame as PandasDF
from sklearn import metrics

__all__ = (
    "precision_given_recall",
    "precision_recall_values",
)


ArrayLike = npt.ArrayLike


def precision_given_recall(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    min_recall: float,
    pos_label: Optional[Union[int, str]] = None,
) -> PandasDF:
    """Compute precision given a desired recall level.

    Examples
    --------
    >>> import onekit.sklearnkit as slk
    >>> import pandas as pd
    >>> y_true = [0, 1, 1, 1, 0, 0, 0, 1]
    >>> y_score = [0.1, 0.4, 0.35, 0.8, 0.5, 0.2, 0.75, 0.5]
    >>> with pd.option_context("display.float_format", "{:.2f}".format):
    ...     slk.precision_given_recall(y_true, y_score, min_recall=0.7)
       threshold  precision  recall
    0       0.40       0.60    0.75
    """
    if not (0 < min_recall <= 1):
        raise ValueError(f"{min_recall=} - must be a float in (0, 1]")
    return (
        precision_recall_values(y_true, y_score, pos_label=pos_label)
        .query(f"recall >= {min_recall}")
        .assign(min_empirical_recall=lambda df: df["recall"].min())
        .query("recall == min_empirical_recall")
        .drop(columns=["min_empirical_recall"])
        .sort_values("precision", ascending=False)
        .reset_index(drop=True)
        .head(1)
    )


def precision_recall_values(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    pos_label: Optional[Union[int, str]] = None,
) -> PandasDF:
    """Compute precision-recall pairs for all thresholds.

    Notes
    -----
    Output of `sklearn.metrics.precision_recall_curve` is wrapped in a DataFrame.

    Examples
    --------
    >>> import onekit.sklearnkit as slk
    >>> import pandas as pd
    >>> y_true = [0, 1, 1, 1, 0, 0, 0, 1]
    >>> y_score = [0.1, 0.4, 0.35, 0.8, 0.5, 0.2, 0.75, 0.5]
    >>> with pd.option_context("display.float_format", "{:.2f}".format):
    ...     slk.precision_recall_values(y_true, y_score)
       threshold  precision  recall
    0       0.10       0.50    1.00
    1       0.20       0.57    1.00
    2       0.35       0.67    1.00
    3       0.40       0.60    0.75
    4       0.50       0.50    0.50
    5       0.75       0.50    0.25
    6       0.80       1.00    0.25
    7        inf       1.00    0.00
    """
    p, r, t = metrics.precision_recall_curve(y_true, y_score, pos_label=pos_label)
    return pd.DataFrame(
        {"threshold": np.append(t, np.inf), "precision": p, "recall": r}
    )
