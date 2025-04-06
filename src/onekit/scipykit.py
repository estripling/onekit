from typing import (
    Iterable,
    NamedTuple,
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
        """Calculate the mean of the Beta distribution."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def mode(self) -> float | None:
        """Calculate the mode of the Beta distribution.

        Note that the mode is undefined for alpha <= 1 or beta <= 1.
        """
        if self.alpha > 1 and self.beta > 1:
            return (self.alpha - 1) / (self.alpha + self.beta - 2)

    def get_summary(self) -> str:
        return pk.concat_strings(
            ", ",
            f"{self} -> mean={pk.num_to_str(self.mean)}",
            f"mode={pk.num_to_str(self.mode)}",
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
    'BetaParams(alpha=4, beta=3) -> mean=0.571429, mode=0.6'

    >>> data = ["head", "tail", "head", "head", "tail"]
    >>> prior = BetaParams(alpha=2, beta=2)
    >>> posterior = sck.compute_beta_posterior(data, prior, pos_label="head")
    >>> posterior.get_summary()
    'BetaParams(alpha=5, beta=4) -> mean=0.555556, mode=0.571429'

    >>> data = [1, 0, 1, 1, 0]
    >>> prior = BetaParams(alpha=1, beta=1)
    >>> posterior1 = sck.compute_beta_posterior(data, prior)
    >>> posterior1.get_summary()
    'BetaParams(alpha=4, beta=3) -> mean=0.571429, mode=0.6'
    >>> more_data = [1, 0, 1, 0, 1]
    >>> posterior2 = sck.compute_beta_posterior(more_data, prior=posterior1)
    >>> posterior2.get_summary()
    'BetaParams(alpha=7, beta=5) -> mean=0.583333, mode=0.6'
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
