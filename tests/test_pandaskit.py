import pandas as pd

import onekit.pandaskit as pdk


def test_union():
    cols = ("a", "b")

    idx1 = (0, 1)
    row11, row12 = (1, 2), (3, 4)
    df1 = pd.DataFrame([row11, row12], idx1, cols)

    idx2 = (2, 3)
    row21, row22 = (5, 6), (7, 8)
    df2 = pd.DataFrame([row21, row22], idx2, cols)

    expected = pd.DataFrame([row11, row12, row21, row22], idx1 + idx2, cols)

    actual = pdk.union(df1, df2)
    assert isinstance(actual, pd.DataFrame)
    pd.testing.assert_frame_equal(actual, expected)
