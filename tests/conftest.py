import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame as PandasDF


@pytest.fixture()
def expected_threshold_summary() -> PandasDF:
    df = pd.DataFrame(
        [
            dict(
                threshold=0.1,
                predicted_positive=8,
                true_positive=4,
                false_positive=4,
                false_negative=0,
                true_negative=0,
                precision=0.5,
                recall=1,
                f1=0.6666666666666666,
                accuracy=0.5,
                balanced_accuracy=0.5,
                matthews_corrcoef=np.nan,
            ),
            dict(
                threshold=0.2,
                predicted_positive=7,
                true_positive=4,
                false_positive=3,
                false_negative=0,
                true_negative=1,
                precision=0.5714285714285714,
                recall=1,
                f1=0.7272727272727273,
                accuracy=0.625,
                balanced_accuracy=0.625,
                matthews_corrcoef=0.3779644730092272,
            ),
            dict(
                threshold=0.35,
                predicted_positive=6,
                true_positive=4,
                false_positive=2,
                false_negative=0,
                true_negative=2,
                precision=0.6666666666666666,
                recall=1,
                f1=0.8,
                accuracy=0.75,
                balanced_accuracy=0.75,
                matthews_corrcoef=0.5773502691896258,
            ),
            dict(
                threshold=0.4,
                predicted_positive=5,
                true_positive=3,
                false_positive=2,
                false_negative=1,
                true_negative=2,
                precision=0.6,
                recall=0.75,
                f1=0.6666666666666665,
                accuracy=0.625,
                balanced_accuracy=0.625,
                matthews_corrcoef=0.2581988897471611,
            ),
            dict(
                threshold=0.5,
                predicted_positive=4,
                true_positive=2,
                false_positive=2,
                false_negative=2,
                true_negative=2,
                precision=0.5,
                recall=0.5,
                f1=0.5,
                accuracy=0.5,
                balanced_accuracy=0.5,
                matthews_corrcoef=0,
            ),
            dict(
                threshold=0.75,
                predicted_positive=2,
                true_positive=1,
                false_positive=1,
                false_negative=3,
                true_negative=3,
                precision=0.5,
                recall=0.25,
                f1=0.3333333333333333,
                accuracy=0.5,
                balanced_accuracy=0.5,
                matthews_corrcoef=0,
            ),
            dict(
                threshold=0.8,
                predicted_positive=1,
                true_positive=1,
                false_positive=0,
                false_negative=3,
                true_negative=4,
                precision=1,
                recall=0.25,
                f1=0.4,
                accuracy=0.625,
                balanced_accuracy=0.625,
                matthews_corrcoef=0.3779644730092272,
            ),
            dict(
                threshold=np.inf,
                predicted_positive=0,
                true_positive=0,
                false_positive=0,
                false_negative=4,
                true_negative=4,
                precision=1,
                recall=0,
                f1=0,
                accuracy=0.5,
                balanced_accuracy=0.5,
                matthews_corrcoef=np.nan,
            ),
        ],
    )

    int_cols = [
        "predicted_positive",
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
    ]

    for int_col in int_cols:
        df[int_col] = df[int_col].astype(int)

    return df
