from typing import (
    Optional,
    Union,
)

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas import DataFrame as PandasDF
from sklearn import metrics
from sklearn.utils import validation

__all__ = (
    "precision_given_recall_score",
    "threshold_summary",
)


ArrayLike = npt.ArrayLike


def precision_given_recall_score(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    min_recall: float,
    pos_label: Optional[Union[int, str]] = None,
) -> float:
    """Compute precision given a desired minimum recall level.

    Examples
    --------
    >>> import onekit.sklearnkit as slk
    >>> y_true = [0, 1, 1, 1, 0, 0, 0, 1]
    >>> y_score = [0.1, 0.4, 0.35, 0.8, 0.5, 0.2, 0.75, 0.5]
    >>> slk.precision_given_recall_score(y_true, y_score, min_recall=0.7)
    0.6
    """
    if not (0 < min_recall <= 1):
        raise ValueError(f"{min_recall=} - must be a float in (0, 1]")

    df = (
        threshold_summary(y_true, y_score, pos_label=pos_label)
        .filter(items=["precision", "recall"])
        .query(f"recall >= {min_recall}")
    )

    min_empirical_recall = df["recall"].min()

    return float(
        0
        if df.empty
        else df.query(f"recall == {min_empirical_recall}")["precision"].max()
    )


def threshold_summary(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    pos_label: Optional[Union[int, str]] = None,
) -> PandasDF:
    """Threshold summary.

    Notes
    -----
    - Support for binary classification only
    - Assumpution: classifier returns scores
    - First values correspond to the edge case where everything is predicted positive
    - Last values correspond to the edge case where everything is predicted negative

    Examples
    --------
    >>> import onekit.sklearnkit as slk
    >>> y_true = [0, 1, 1, 1, 0, 0, 0, 1]
    >>> y_score = [0.1, 0.4, 0.35, 0.8, 0.5, 0.2, 0.75, 0.5]
    >>> with pd.option_context("display.float_format", "{:.2f}".format):
    ...     slk.threshold_summary(y_true, y_score).T
                          0    1    2    3    4    5    6    7
    threshold          0.10 0.20 0.35 0.40 0.50 0.75 0.80  inf
    predicted_positive 8.00 7.00 6.00 5.00 4.00 2.00 1.00 0.00
    true_positive      4.00 4.00 4.00 3.00 2.00 1.00 1.00 0.00
    false_positive     4.00 3.00 2.00 2.00 2.00 1.00 0.00 0.00
    false_negative     0.00 0.00 0.00 1.00 2.00 3.00 3.00 4.00
    true_negative      0.00 1.00 2.00 2.00 2.00 3.00 4.00 4.00
    precision          0.50 0.57 0.67 0.60 0.50 0.50 1.00 1.00
    recall             1.00 1.00 1.00 0.75 0.50 0.25 0.25 0.00
    f1                 0.67 0.73 0.80 0.67 0.50 0.33 0.40 0.00
    accuracy           0.50 0.62 0.75 0.62 0.50 0.50 0.62 0.50
    balanced_accuracy  0.50 0.62 0.75 0.62 0.50 0.50 0.62 0.50
    matthews_corrcoef   NaN 0.38 0.58 0.26 0.00 0.00 0.38  NaN
    """
    y = validation.column_or_1d(y_true)
    s = validation.column_or_1d(y_score)
    validation.check_consistent_length(y, s)
    validation.assert_all_finite(y)
    validation.assert_all_finite(s)
    pos_label = validation._check_pos_label_consistency(pos_label, y)

    precision, recall, thresholds = metrics.precision_recall_curve(
        y_true=y,
        y_score=s,
        pos_label=pos_label,
        sample_weight=None,
        drop_intermediate=False,
    )

    is_true_pos = y == pos_label
    is_true_neg = y != pos_label

    def is_pred_pos(t: float) -> np.ndarray:
        return s >= t

    def is_pred_neg(t: float) -> np.ndarray:
        return s < t

    return (
        pd.DataFrame(np.append(thresholds, np.inf), columns=["t"])
        .assign(
            pp=lambda df: df.t.map(lambda t: is_pred_pos(t).sum()),
            tp=lambda df: df.t.map(lambda t: (is_pred_pos(t) & is_true_pos).sum()),
            fp=lambda df: df.t.map(lambda t: (is_pred_pos(t) & is_true_neg).sum()),
            fn=lambda df: df.t.map(lambda t: (is_pred_neg(t) & is_true_pos).sum()),
            tn=lambda df: df.t.map(lambda t: (is_pred_neg(t) & is_true_neg).sum()),
            precision=precision,
            recall=recall,
            f1=2 * (precision * recall) / (precision + recall),
            acc=lambda df: (df.tp + df.tn) / (df.tp + df.tn + df.fp + df.fn),
            bacc=lambda df: 0.5 * (df.tp / (df.tp + df.fn) + df.tn / (df.tn + df.fp)),
            mcc=lambda df: np.true_divide(
                (df.tp * df.tn - df.fp * df.fn),
                np.sqrt(
                    (df.tp + df.fp)
                    * (df.tp + df.fn)
                    * (df.tn + df.fp)
                    * (df.tn + df.fn)
                ),
            ),
        )
        .rename(
            columns={
                "t": "threshold",
                "pp": "predicted_positive",
                "tp": "true_positive",
                "fp": "false_positive",
                "fn": "false_negative",
                "tn": "true_negative",
                "acc": "accuracy",
                "bacc": "balanced_accuracy",
                "mcc": "matthews_corrcoef",
            },
        )
    )
