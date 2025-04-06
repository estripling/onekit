import numpy.testing as npt
import pytest

from onekit import scipykit as sck
from onekit.scipykit import BetaParams


@pytest.mark.parametrize(
    "beta_params, expected",
    [
        (
            BetaParams(alpha=0.5, beta=0.5),
            {
                "alpha": 0.5,
                "beta": 0.5,
                "mean": 0.5 / (0.5 + 0.5),
                "mode": None,
                "hdi": None,
            },
        ),
        (
            BetaParams(alpha=1, beta=1),
            {"alpha": 1, "beta": 1, "mean": 1 / (1 + 1), "mode": None, "hdi": None},
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
        (
            BetaParams(alpha=100, beta=100),
            {
                "alpha": 100,
                "beta": 100,
                "mean": 0.5,
                "mode": 0.5,
                "hdi": (0.430951, 0.569049),
            },
        ),
        (
            BetaParams(alpha=117, beta=103),
            {
                "alpha": 117,
                "beta": 103,
                "mean": 117 / (117 + 103),
                "mode": (117 - 1) / (117 + 103 - 2),
                "hdi": (0.466015, 0.59746),
            },
        ),
        (
            BetaParams(alpha=18.25, beta=6.75),
            {
                "alpha": 18.25,
                "beta": 6.75,
                "mean": 18.25 / (18.25 + 6.75),
                "mode": (18.25 - 1) / (18.25 + 6.75 - 2),
                "hdi": (0.558194, 0.891582),
            },
        ),
        (
            BetaParams(alpha=35.25, beta=9.75),
            {
                "alpha": 35.25,
                "beta": 9.75,
                "mean": 35.25 / (35.25 + 9.75),
                "mode": (35.25 - 1) / (35.25 + 9.75 - 2),
                "hdi": (0.662908, 0.896649),
            },
        ),
        (
            BetaParams(alpha=18, beta=4),
            {
                "alpha": 18,
                "beta": 4,
                "mean": 18 / (18 + 4),
                "mode": (18 - 1) / (18 + 4 - 2),
                "hdi": (0.659947, 0.959123),
            },
        ),
    ],
)
def test_beta_params(beta_params: BetaParams, expected: dict[str, int | float]):
    for prop, expected_value in expected.items():
        attr = getattr(beta_params, prop)
        actual_value = attr() if callable(attr) else attr
        if isinstance(actual_value, tuple):
            npt.assert_array_almost_equal(actual_value, expected_value)
        else:
            assert actual_value == expected_value


@pytest.mark.parametrize(
    "beta_params, hdi_prob, expected",
    [
        (BetaParams(alpha=2, beta=2), 0.95, (0.094299, 0.905701)),
        (BetaParams(alpha=2, beta=2), 0.90, (0.13535, 0.86465)),
        (BetaParams(alpha=4, beta=4), 0.95, (0.184052, 0.815948)),
        (BetaParams(alpha=4, beta=4), 0.90, (0.225322, 0.774678)),
        (BetaParams(alpha=100, beta=100), 0.95, (0.430951, 0.569049)),
        (BetaParams(alpha=100, beta=100), 0.90, (0.44197, 0.55803)),
    ],
)
def test_beta_params__hdi(
    beta_params: BetaParams,
    hdi_prob: float,
    expected: tuple[float, float],
):
    actual = beta_params.hdi(hdi_prob)
    npt.assert_array_almost_equal(actual, expected)


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
