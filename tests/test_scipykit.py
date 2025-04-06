import numpy.testing as npt
import pytest

from onekit import scipykit as sck
from onekit.scipykit import BetaParams


@pytest.mark.parametrize(
    "beta_params, expected",
    [
        (
            BetaParams(alpha=0.5, beta=0.5),
            {"alpha": 0.5, "beta": 0.5, "mean": 0.5, "mode": None, "hdi": None},
        ),
        (
            BetaParams(alpha=1, beta=1),
            {"alpha": 1, "beta": 1, "mean": 0.5, "mode": None, "hdi": None},
        ),
        (
            BetaParams(alpha=2, beta=2),
            {
                "alpha": 2,
                "beta": 2,
                "mean": 0.5,
                "mode": 0.5,
                "hdi": (0.094299, 0.905701),
            },
        ),
        (
            BetaParams(alpha=4, beta=3),
            {
                "alpha": 4,
                "beta": 3,
                "mean": 4 / 7,
                "mode": 0.6,
                "hdi": (0.238706, 0.895169),
            },
        ),
        (
            BetaParams(alpha=5, beta=4),
            {
                "alpha": 5,
                "beta": 4,
                "mean": 5 / 9,
                "mode": 4 / 7,
                "hdi": (0.254086, 0.851192),
            },
        ),
        (
            BetaParams(alpha=7, beta=5),
            {
                "alpha": 7,
                "beta": 5,
                "mean": 7 / 12,
                "mode": 6 / 10,
                "hdi": (0.318232, 0.841428),
            },
        ),
    ],
)
def test_compute_beta_params(beta_params: BetaParams, expected: dict[str, int | float]):
    for prop, expected_value in expected.items():
        attr = getattr(beta_params, prop)
        actual_value = attr() if callable(attr) else attr
        if isinstance(actual_value, tuple):
            npt.assert_array_almost_equal(actual_value, expected_value)
        else:
            assert actual_value == expected_value


@pytest.mark.parametrize(
    "data, prior, expected",
    [
        ([1, 0, 1, 1, 0], BetaParams(alpha=1, beta=1), BetaParams(alpha=4, beta=3)),
        ([1, 0, 1, 1, 0], BetaParams(alpha=2, beta=2), BetaParams(alpha=5, beta=4)),
    ],
)
def test_compute_beta_posterior(
    data: list[int],
    prior: BetaParams,
    expected: BetaParams,
):
    actual = sck.compute_beta_posterior(data, prior)
    assert isinstance(actual, BetaParams)
    for prop in ["alpha", "beta", "mean", "mode"]:
        actual_value = getattr(actual, prop)
        expected_value = getattr(expected, prop)
        assert actual_value == expected_value
