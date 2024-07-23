import numpy as np
import pandas as pd
from pandas import DataFrame as PandasDF
from sklearn import metrics

import onekit.sklearnkit as slk


def test_threshold_summary(expected_threshold_summary: PandasDF):
    y_true = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8, 0.5, 0.2, 0.75, 0.5])

    actual = slk.threshold_summary(y_true, y_score)
    pd.testing.assert_frame_equal(actual, expected_threshold_summary)

    for t in expected_threshold_summary["threshold"]:
        y_pred = y_score >= t

        actual_scores = (
            actual.query(f"threshold == {t}")
            .filter(
                [
                    "precision",
                    "recall",
                    "f1",
                    "accuracy",
                    "balanced_accuracy",
                    "matthews_corrcoef",
                ],
            )
            .fillna(0)
            .reset_index(drop=True)
            .T
        )

        expected_scores = pd.Series(
            {
                # precision_recall_curve assigns 1 instead of 0 for precision
                "precision": (
                    1.0 if np.isinf(t) else metrics.precision_score(y_true, y_pred)
                ),
                "recall": metrics.recall_score(y_true, y_pred),
                "f1": metrics.f1_score(y_true, y_pred),
                "accuracy": metrics.accuracy_score(y_true, y_pred),
                "balanced_accuracy": metrics.balanced_accuracy_score(y_true, y_pred),
                "matthews_corrcoef": metrics.matthews_corrcoef(y_true, y_pred),
            }
        ).to_frame()

        pd.testing.assert_frame_equal(actual_scores, expected_scores)
