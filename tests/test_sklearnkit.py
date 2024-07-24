import numpy as np
import pandas as pd
import pytest

import onekit.sklearnkit as slk


class TestPrecisionRecall:
    def test_precision_given_recall(self, y_true: np.ndarray, y_score: np.ndarray):
        actual = slk.precision_given_recall(y_true, y_score, min_recall=0.7)
        expected = pd.DataFrame([dict(threshold=0.4, precision=0.6, recall=0.75)])
        pd.testing.assert_frame_equal(actual, expected)

        with pytest.raises(ValueError):
            slk.precision_given_recall(y_true, y_score, min_recall=0)

    def test_precision_given_recall_score(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
    ):
        actual = slk.precision_given_recall_score(y_true, y_score, min_recall=0.7)
        expected = 0.6

        assert isinstance(actual, float)
        assert actual == expected

    def test_precision_given_recall_summary(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
    ):
        actual = slk.precision_given_recall_summary(y_true, y_score, min_recall=0.7)
        expected_data = dict(
            threshold=0.4,
            predicted_positive=5,
            true_positive=3,
            false_positive=2,
            false_negative=1,
            true_negative=2,
            precision=0.6,
            recall=0.75,
            f1=2 / 3,
            accuracy=0.625,
            balanced_accuracy=0.625,
            matthews_corrcoef=0.2581988897471611,
            pos_lr=1.5,
            neg_lr=0.5,
        )
        expected = pd.DataFrame([expected_data])
        pd.testing.assert_frame_equal(actual, expected)

    def test_precision_recall_values(self, y_true: np.ndarray, y_score: np.ndarray):
        actual = slk.precision_recall_values(y_true, y_score)
        expected = pd.DataFrame(
            [
                dict(threshold=0.1, precision=0.5, recall=1.0),
                dict(threshold=0.2, precision=4 / 7, recall=1.0),
                dict(threshold=0.35, precision=2 / 3, recall=1.0),
                dict(threshold=0.4, precision=0.6, recall=0.75),
                dict(threshold=0.5, precision=0.5, recall=0.5),
                dict(threshold=0.75, precision=0.5, recall=0.25),
                dict(threshold=0.8, precision=1.0, recall=0.25),
                dict(threshold=np.inf, precision=1.0, recall=0.0),
            ]
        )
        pd.testing.assert_frame_equal(actual, expected)

    @pytest.fixture()
    def y_true(self) -> np.ndarray:
        return np.array([0, 1, 1, 1, 0, 0, 0, 1])

    @pytest.fixture()
    def y_score(self) -> np.ndarray:
        return np.array([0.1, 0.4, 0.35, 0.8, 0.5, 0.2, 0.75, 0.5])
