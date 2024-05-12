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
    "x, kind, expected",
    [
        (
            [],
            "log",
            np.array([]),
        ),
        (
            [0.1, 1, 10, 100, 1_000, 10_000, 100_000, 1_000_000],
            "log",
            np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        ),
        (
            (0.2, 2, 20),
            "log",
            np.array([0.30102999566398125, 1.3010299956639813, 2.3010299956639813]),
        ),
        (
            np.array([-0.5, -5, -50]),
            "log",
            np.array([0.6989700043360187, 1.6989700043360187, 2.6989700043360187]),
        ),
        (
            np.array([np.inf, np.nan, 20.0, 10_000_000.0]),
            "log",
            np.array([np.inf, 0.0, 2.3010299956639813, 8.0]),
        ),
        (
            (0.2, 2, 20),
            "int",
            np.array([0, 1, 2]),
        ),
        (
            np.array([-0.5, -5, -50]),
            "int",
            np.array([0, 1, 2]),
        ),
        (
            (0.1, 1, 10, 100, 1_000, 10_000, 100_000, 1_000_000),
            "linear",
            np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        ),
        (
            (0.2, 2, 20),
            "linear",
            np.array([0.11111111111111112, 1.1111111111111112, 2.111111111111111]),
        ),
    ],
)
def test_digitscale(x: np.ndarray, kind: str, expected: np.ndarray):
    actual = npk.digitscale(x, kind=kind)
    npt.assert_array_equal(actual, expected)


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
