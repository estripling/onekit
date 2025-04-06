import operator
from typing import (
    Iterable,
    NamedTuple,
)

from scipy import (
    optimize,
    stats,
)

from onekit import numpykit as npk
from onekit import pythonkit as pk

__all__ = (
    "BetaParams",
    "compute_beta_posterior",
)


class BetaParams(NamedTuple):
    """Represents the parameters of a Beta distribution."""

    alpha: int | float = 1
    beta: int | float = 1

    @property
    def mean(self) -> float:
        """Compute the mean of the Beta distribution."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def mode(self) -> float | None:
        """Compute the mode of the Beta distribution.

        Note that the mode is undefined for alpha <= 1 or beta <= 1.
        """
        if self.alpha > 1 and self.beta > 1:
            return (self.alpha - 1) / (self.alpha + self.beta - 2)

    def hdi(self, hdi_prob: float = 0.95) -> tuple[float, float] | None:
        """Compute the highest density interval (HDI) of the Beta distribution.

        Note that the HDI is not computed for alpha <= 1 or beta <= 1.
        """
        if self.alpha > 1 and self.beta > 1:
            beta_dist = stats.beta(self.alpha, self.beta)
            tail_prob = 1 - hdi_prob

            def interval_width(x: float) -> float:
                return beta_dist.ppf(hdi_prob + x) - beta_dist.ppf(x)

            hdi_tail_prob = operator.getitem(
                optimize.fmin(interval_width, tail_prob, ftol=1e-12, disp=False),
                0,
            )
            hdi_endpoints = beta_dist.ppf([hdi_tail_prob, hdi_prob + hdi_tail_prob])
            return hdi_endpoints[0], hdi_endpoints[1]

    def get_summary(self, hdi_prob: float = 0.95) -> str:
        """Compute summary statistics of the Beta distribution."""
        mode = self.mode
        mode_info = f"mode={pk.num_to_str(mode)}" if mode is not None else None

        hdi_info = None
        hdi_endpoints = self.hdi(hdi_prob)
        if hdi_endpoints is not None:
            hdi_pct = pk.num_to_str(100 * hdi_prob)
            hdi_lower_endpoint, hdi_upper_endpoint = map(pk.num_to_str, hdi_endpoints)
            hdi_info = f"{hdi_pct}%-HDI=[{hdi_lower_endpoint}, {hdi_upper_endpoint}]"

        return pk.concat_strings(
            " ",
            f"{self} ->",
            f"mean={pk.num_to_str(self.mean)}",
            mode_info,
            hdi_info,
        )


def compute_beta_posterior(
    data: Iterable[int | str],
    prior: BetaParams | None = None,
    pos_label: int | str = 1,
) -> BetaParams:
    """Update Beta prior with observed binomial data to compute posterior.

    This function applies Bayesian inference to update the parameters of a Beta
    distribution, given observed binomial data. The Beta distribution is commonly used
    as a prior in binomial proportion estimation due to its conjugacy, simplifying the
    calculation of the posterior.

    Examples
    --------
    >>> from onekit import scipykit as sck
    >>> from onekit.scipykit import BetaParams
    >>> data = [1, 0, 1, 1, 0]
    >>> posterior = sck.compute_beta_posterior(data)
    >>> posterior.get_summary()
    'BetaParams(alpha=4, beta=3) -> mean=0.571429 mode=0.6 95%-HDI=[0.238706, 0.895169]'

    >>> data = ["head", "tail", "head", "head", "tail", "head", "head", "tail"]
    >>> prior = BetaParams(alpha=2, beta=2)
    >>> posterior = sck.compute_beta_posterior(data, prior, pos_label="head")
    >>> posterior.get_summary()
    'BetaParams(alpha=7, beta=5) -> mean=0.583333 mode=0.6 95%-HDI=[0.318232, 0.841428]'

    >>> data = [1, 0, 1, 1, 0]
    >>> prior = BetaParams(alpha=1, beta=1)
    >>> posterior1 = sck.compute_beta_posterior(data, prior)
    >>> posterior1.get_summary()
    'BetaParams(alpha=4, beta=3) -> mean=0.571429 mode=0.6 95%-HDI=[0.238706, 0.895169]'
    >>> more_data = [1, 0, 1, 0, 1]
    >>> posterior2 = sck.compute_beta_posterior(more_data, prior=posterior1)
    >>> posterior2.get_summary()
    'BetaParams(alpha=7, beta=5) -> mean=0.583333 mode=0.6 95%-HDI=[0.318232, 0.841428]'
    """
    prior = prior or BetaParams()
    y = npk.create_boolean_array(data, pos_label)
    num_successes = y.sum()
    num_trials = len(y)
    posterior = BetaParams(
        alpha=prior.alpha + num_successes,
        beta=prior.beta + num_trials - num_successes,
    )
    return posterior
