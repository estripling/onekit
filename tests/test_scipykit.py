import pytest

from onekit import scipykit as sck
from onekit.scipykit import BetaParams


@pytest.mark.parametrize(
    "beta_params, expected",
    [
        (
            BetaParams(alpha=1, beta=1),
            {"alpha": 1, "beta": 1, "mean": 0.5, "mode": None},
        ),
        (
            BetaParams(alpha=2, beta=2),
            {"alpha": 2, "beta": 2, "mean": 0.5, "mode": 0.5},
        ),
        (
            BetaParams(alpha=4, beta=3),
            {"alpha": 4, "beta": 3, "mean": 0.5714285714285714, "mode": 0.6},
        ),
    ],
)
def test_compute_beta_params(beta_params: BetaParams, expected: dict[str, int | float]):
    for prop, expected_value in expected.items():
        actual_value = getattr(beta_params, prop)
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
