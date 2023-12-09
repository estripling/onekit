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
