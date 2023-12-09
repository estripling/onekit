import pandas as pd

import onekit.pandaskit as pdk


def test_join():
    idx = (0, 1)

    cols1 = ("a", "b")
    row11, row12 = (1, 3), (2, 4)
    df1 = pd.DataFrame([row11, row12], idx, cols1)

    cols2 = ("a", "c")
    row21, row22 = (1, 5), (2, 6)
    df2 = pd.DataFrame([row21, row22], idx, cols2)

    actual = pdk.join(df1, df2, on="a")
    assert isinstance(actual, pd.DataFrame)

    expected = pd.merge(df1, df2, on="a")
    pd.testing.assert_frame_equal(actual, expected)


def test_profile():
    data = {
        "a": [True, None, False, False, True, False],
        "b": [1] * 6,
        "c": [None] * 6,
    }
    df = pd.DataFrame(data)

    actual = pdk.profile(df)
    assert isinstance(actual, pd.DataFrame)

    expected = pd.DataFrame(
        {
            "type": {"a": "object", "b": "int64", "c": "object"},
            "count": {"a": 5, "b": 6, "c": 0},
            "isnull": {"a": 1, "b": 0, "c": 6},
            "isnull%": {"a": 0.16666666666666, "b": 0.0, "c": 1.0},
            "unique": {"a": 2, "b": 1, "c": 0},
            "unique%": {"a": 0.33333333333, "b": 0.166666666666, "c": 0.0},
            "mean": {"a": float("nan"), "b": 1.0, "c": float("nan")},
            "std": {"a": float("nan"), "b": 0.0, "c": float("nan")},
            "skewness": {"a": float("nan"), "b": 0.0, "c": float("nan")},
            "kurtosis": {"a": float("nan"), "b": 0.0, "c": float("nan")},
            "min": {"a": float("nan"), "b": 1.0, "c": float("nan")},
            "5%": {"a": float("nan"), "b": 1.0, "c": float("nan")},
            "25%": {"a": float("nan"), "b": 1.0, "c": float("nan")},
            "50%": {"a": float("nan"), "b": 1.0, "c": float("nan")},
            "75%": {"a": float("nan"), "b": 1.0, "c": float("nan")},
            "95%": {"a": float("nan"), "b": 1.0, "c": float("nan")},
            "max": {"a": float("nan"), "b": 1.0, "c": float("nan")},
        }
    )

    pd.testing.assert_frame_equal(actual, expected)


def test_union():
    cols = ("a", "b")

    idx1 = (0, 1)
    row11, row12 = (1, 2), (3, 4)
    df1 = pd.DataFrame([row11, row12], idx1, cols)

    idx2 = (2, 3)
    row21, row22 = (5, 6), (7, 8)
    df2 = pd.DataFrame([row21, row22], idx2, cols)

    actual = pdk.union(df1, df2)
    assert isinstance(actual, pd.DataFrame)

    expected = pd.DataFrame([row11, row12, row21, row22], idx1 + idx2, cols)
    pd.testing.assert_frame_equal(actual, expected)
