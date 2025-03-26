import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas import DataFrame as PandasDF
from sklearn import metrics

__all__ = (
    "precision_given_recall",
    "precision_given_recall_score",
    "precision_given_recall_summary",
    "precision_recall_values",
)


ArrayLike = npt.ArrayLike


def precision_given_recall(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    min_recall: float,
    pos_label: int | str | None = None,
) -> PandasDF:
    """Compute precision given a desired recall level.

    Examples
    --------
    >>> from onekit import sklearnkit as slk
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


def precision_given_recall_score(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    min_recall: float,
    pos_label: int | str | None = None,
) -> float:
    """Compute precision score given a desired recall level.

    Examples
    --------
    >>> import numpy as np
    >>> from onekit import sklearnkit as slk
    >>> np.set_printoptions(legacy="1.21")
    >>> y_true = [0, 1, 1, 1, 0, 0, 0, 1]
    >>> y_score = [0.1, 0.4, 0.35, 0.8, 0.5, 0.2, 0.75, 0.5]
    >>> slk.precision_given_recall_score(y_true, y_score, min_recall=0.7)
    0.6
    """
    return precision_given_recall(
        y_true,
        y_score,
        min_recall=min_recall,
        pos_label=pos_label,
    ).at[0, "precision"]


def precision_given_recall_summary(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    min_recall: float,
    pos_label: int | str | None = None,
) -> PandasDF:
    """Extend `precision_given_recall` with additional performance metrics.

    Examples
    --------
    >>> from onekit import sklearnkit as slk
    >>> import pandas as pd
    >>> y_true = [0, 1, 1, 1, 0, 0, 0, 1]
    >>> y_score = [0.1, 0.4, 0.35, 0.8, 0.5, 0.2, 0.75, 0.5]
    >>> with pd.option_context("display.float_format", "{:.4f}".format):
    ...     slk.precision_given_recall_summary(y_true, y_score, min_recall=0.7).T
                            0
    threshold          0.4000
    predicted_positive 5.0000
    true_positive      3.0000
    false_positive     2.0000
    false_negative     1.0000
    true_negative      2.0000
    precision          0.6000
    recall             0.7500
    f1                 0.6667
    accuracy           0.6250
    balanced_accuracy  0.6250
    matthews_corrcoef  0.2582
    pos_lr             1.5000
    neg_lr             0.5000
    """
    pr = precision_given_recall(
        y_true,
        y_score,
        min_recall=min_recall,
        pos_label=pos_label,
    )
    t = pr.at[0, "threshold"]
    y_pred = y_score >= t
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()

    pos_label = 1 if pos_label is None else pos_label
    neg_label = set(y_true).difference({pos_label}).pop()
    labels = [neg_label, pos_label]
    pos_lr, neg_lr = metrics.class_likelihood_ratios(y_true, y_pred, labels=labels)

    data = dict(
        threshold=t,
        predicted_positive=tp + fp,
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        true_negative=tn,
        precision=pr.at[0, "precision"],
        recall=pr.at[0, "recall"],
        f1=metrics.f1_score(y_true, y_pred),
        accuracy=metrics.accuracy_score(y_true, y_pred),
        balanced_accuracy=metrics.balanced_accuracy_score(y_true, y_pred),
        matthews_corrcoef=metrics.matthews_corrcoef(y_true, y_pred),
        pos_lr=pos_lr,
        neg_lr=neg_lr,
    )

    return pd.DataFrame([data])


def precision_recall_values(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    pos_label: int | str | None = None,
) -> PandasDF:
    """Compute precision-recall pairs for all thresholds.

    Notes
    -----
    Output of `sklearn.metrics.precision_recall_curve` is wrapped in a DataFrame.

    Examples
    --------
    >>> from onekit import sklearnkit as slk
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
