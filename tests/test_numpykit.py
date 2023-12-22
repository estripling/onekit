import numpy as np
import numpy.testing as npt
import pytest

import onekit.numpykit as npk


def test_check_vector():
    x = [1, 2, 3]
    actual = npk.check_vector(x)
    npt.assert_array_equal(actual, np.array(x))

    with pytest.raises(
        TypeError,
        match=r"input must be a vector-like object - it has shape=\(1, 2\)",
    ):
        npk.check_vector([[1, 2]])

    with pytest.raises(
        TypeError,
        match=r"x with n=2 - n must be an integer in \[3, inf\)",
    ):
        npk.check_vector([1, 2], n_min=3)

    with pytest.raises(
        TypeError,
        match=r"x with n=1 - n must be an integer in \[2, 2\]",
    ):
        npk.check_vector([1], n_min=2, n_max=2)


@pytest.mark.parametrize(
    "x, expected",
    [
        ([], np.nan),
        ([np.nan], np.nan),
        ([np.nan, np.nan], np.nan),
        ([1], np.nan),
        ([1, 1], 0.0),
        ([98, 127, 82, 67, 121, np.nan, 119, 92, 110, 113, 107], 5.963220606350229),
    ],
)
def test_stderr(x: npk.ArrayLike, expected: float):
    actual = npk.stderr(x)
    npt.assert_array_equal(actual, expected)
