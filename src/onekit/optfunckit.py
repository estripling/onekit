"""Single-objective test functions for minimization. [wiki]_

References
----------
.. [wiki] "Test functions for optimization", Wikipedia,
    `<https://en.wikipedia.org/wiki/Test_functions_for_optimization>`_
"""
from typing import (
    Callable,
    List,
    NamedTuple,
    Optional,
)

import numpy as np
import numpy.typing as npt
import toolz

__all__ = (
    "ackley",
    "beale",
    "bump",
    "check_vector",
    "fetch_minima",
    "negate",
    "peaks",
    "rastrigin",
    "rosenbrock",
    "schwefel",
    "sinc",
    "sphere",
)

Vector = npt.NDArray[np.float64]


class Minimum(NamedTuple):
    """Define minimum for :math:`f\\colon \\mathbb{R}^{n} \\rightarrow \\mathbb{R}`."""

    x: Vector
    fx: float

    @property
    def n(self) -> int:
        """Dimensionality of :math:`x`."""
        return len(self.x)


def ackley(x: Vector, /) -> float:
    """Ackley function.

    A function :math:`f\\colon \\mathbb{R}^{n} \\rightarrow \\mathbb{R}`
    that takes an :math:`n`-vector as input and returns a scalar value.
    [wiki]_

    .. math::

        f(\\mathbf{x}) =
        -20 \\exp \\left(
            -0.2 \\sqrt{ \\frac{1}{n} \\sum_{i=1}^{n} x_i^2 }
        \\right)
        - \\exp \\left( \\frac{1}{n} \\sum_{i=1}^{n} \\cos(2 \\pi x_i) \\right)
        + 20
        + e

    Examples
    --------
    >>> import onekit.optfunckit as ofk
    >>> round(ofk.ackley([0, 0]), 4)
    0.0

    >>> round(ofk.ackley([1, 2]), 4)
    5.4221

    >>> round(ofk.ackley([1, 2, 3]), 4)
    7.0165
    """
    x = check_vector(x)
    return float(
        -20 * np.exp(-0.2 * np.sqrt((x**2).mean()))
        - np.exp((np.cos(2 * np.pi * x)).sum() / len(x))
        + 20
        + np.e
    )


def beale(x: Vector, /) -> float:
    """Beale function.

    A function :math:`f\\colon \\mathbb{R}^{2} \\rightarrow \\mathbb{R}`
    that takes a :math:`2`-vector as input and returns a scalar value.
    [wiki]_ [beale]_

    .. math::

       f(\\mathbf{x}) =
       \\left( 1.5 - x_{1} + x_{1} x_{2} \\right)^{2}
       + \\left( 2.25 - x_{1} + x_{1} x_{2}^{2} \\right)^{2}
       + \\left( 2.625 - x_{1} + x_{1} x_{2}^{3}\\right)^{2}

    References
    ----------
    .. [beale] "Beale function", Virtual Library of Simulation Experiments:
        Test Functions and Datasets, `<https://www.sfu.ca/~ssurjano/beale.html>`_

    Examples
    --------
    >>> import onekit.optfunckit as ofk
    >>> ofk.beale([3, 0.5])
    0.0

    >>> round(ofk.beale([0, 0]), 4)
    14.2031

    >>> round(ofk.beale([1, 1]), 4)
    14.2031

    >>> round(ofk.beale([2, 2]), 4)
    356.7031
    """
    x1, x2 = check_vector(x, n_min=2, n_max=2)
    f1 = (1.5 - x1 + x1 * x2) ** 2
    f2 = (2.25 - x1 + x1 * x2**2) ** 2
    f3 = (2.625 - x1 + x1 * x2**3) ** 2
    return float(f1 + f2 + f3)


def bump(x: Vector, /) -> float:
    """Bump function.

    A function :math:`f\\colon \\mathbb{R}^{n} \\rightarrow \\mathbb{R}`
    that takes an :math:`n`-vector as input and returns a scalar value.
    [bump]_

    .. math::

        f(\\mathbf{x}) =
        \\begin{cases}
            -\\exp\\left(-\\frac{1}{1 - r^{2}}\\right)
              & \\text{ if } r = ||\\mathbf{x}|| < 1 \\\\
            0 & \\text{ otherwise }
        \\end{cases}

    References
    ----------
    .. [bump] "Bump function", Wikipedia,
        `<https://en.wikipedia.org/wiki/Bump_function>`_

    Examples
    --------
    >>> import onekit.optfunckit as ofk
    >>> round(ofk.bump([0, 0]), 4)
    -0.3679

    >>> round(ofk.bump([0.5, 0.5]), 4)
    -0.1353

    >>> ofk.bump([1, 1])
    0.0
    """
    x = check_vector(x)
    r = np.sqrt((x**2).sum())
    return negate(np.exp(-1 / (1 - r**2)) if r < 1 else 0)


def check_vector(x: np.array, /, *, n_min: int = 1, n_max: int = np.inf) -> Vector:
    """Validate :math:`n`-vector.

    Parameters
    ----------
    x : array_like
        The input object to be validated to represent an :math:`n`-vector.
    n_min : int, default=1
        Specify the minimum number of :math:`n`.
    n_max : int, default=inf
        Specify the maximum number of :math:`n`.

    Raises
    ------
    TypeError
        - If ``x`` is not vector-like.
        - If ``n`` is not between ``n_min`` and ``n_max``.

    Examples
    --------
    >>> import onekit.optfunckit as ofk
    >>> ofk.check_vector([0, 0])
    array([0, 0])
    """
    n_max = n_max or np.inf
    x = np.atleast_1d(x)

    if len(x.shape) != 1:
        raise TypeError(f"input must be a vector-like object - it has shape={x.shape}")

    if not (n_min <= len(x) <= n_max):
        domain = f"[{n_min}, {n_max}"
        domain = f"{domain}]" if np.isfinite(n_max) else f"{domain})"
        raise TypeError(f"x with n={len(x)} - n must be an integer in {domain}")

    return x


@toolz.curry
def fetch_minima(func: Callable, /, n: int) -> Optional[List[Minimum]]:
    """Get minima for defined functions.

    Examples
    --------
    >>> import onekit.optfunckit as ofk
    >>> sphere_minima_n5 = ofk.fetch_minima(ofk.sphere, 5)
    >>> sphere_minima_n5
    [Minimum(x=array([0, 0, 0, 0, 0]), fx=0)]
    >>> minimum = sphere_minima_n5[0]
    >>> minimum.n
    5
    """
    minima = {
        ackley: [Minimum(check_vector([0] * n), 0)],
        beale: [Minimum(check_vector([3, 0.5]), 0)],
        bump: [Minimum(check_vector([0] * n), -0.36787944117144233)],
        peaks: [
            Minimum(
                check_vector([0.228279999979237, -1.625531071954464]),
                -6.551133332622496,
            )
        ],
        rastrigin: [Minimum(check_vector([0] * n), 0)],
        rosenbrock: [Minimum(check_vector([1] * n), 0)],
        schwefel: [Minimum(check_vector([420.9687] * n), 0)],
        sinc: [
            Minimum(
                check_vector([-4.493409471849579]),
                -0.217233628211222,
            ),
            Minimum(
                check_vector([4.493409471849579]),
                -0.217233628211222,
            ),
        ],
        sphere: [Minimum(check_vector([0] * n), 0)],
    }
    return minima.get(func, None)


def negate(fx: float) -> float:
    """Change sign of real number.

    By convention, the standard form for an optimization problem defines
    a minimization problem. A maximization problem can be treated by negating
    the objective function.
    [opt]_

    References
    ----------
    .. [opt] "Optimization problem", Wikipedia,
        `<https://en.wikipedia.org/wiki/Optimization_problem>`_

    Examples
    --------
    >>> import onekit.optfunckit as ofk
    >>> ofk.negate(1.0)
    -1.0
    """
    return 0.0 if np.isclose(fx, 0) else -float(fx)


def peaks(x: Vector, /) -> float:
    """Peaks function.

    A function :math:`f\\colon \\mathbb{R}^{2} \\rightarrow \\mathbb{R}`
    that takes a :math:`2`-vector as input and returns a scalar value.

    .. math::

       f(\\mathbf{x}) =
       3 (1 - x_{1})^{2}
         \\exp\\left( - x_{1}^{2} - (x_{2} + 1)^{2} \\right)
       - 10 \\left( \\frac{x_{1}}{5} - x_{1}^{3} - x_{2}^{5} \\right)
         \\exp\\left( - x_{1}^{2} - x_{2}^{2} \\right)
       - \\frac{1}{3} \\exp\\left( - (x_{1} + 1)^{2} - x_{2}^{2} \\right)

    Examples
    --------
    >>> import onekit.optfunckit as ofk
    >>> round(ofk.peaks([0, 0]), 4)
    0.981
    """
    x1, x2 = check_vector(x, n_min=2, n_max=2)
    f1 = 3 * (1 - x1) ** 2 * np.exp(-(x1**2) - (x2 + 1) ** 2)
    f2 = 10 * (x1 / 5 - x1**3 - x2**5) * np.exp(-(x1**2) - x2**2)
    f3 = 1 / 3 * np.exp(-((x1 + 1) ** 2) - x2**2)
    return float(f1 - f2 - f3)


def rastrigin(x: Vector, /) -> float:
    """Rastrigin function.

    A function :math:`f\\colon \\mathbb{R}^{n} \\rightarrow \\mathbb{R}`
    that takes an :math:`n`-vector as input and returns a scalar value.
    [wiki]_

    .. math::

        f(\\mathbf{x}) =
        10n + \\sum_{i=1}^{n} \\left( x_i^2 - 10 \\cos(2 \\pi x_i) \\right)

    Examples
    --------
    >>> import onekit.optfunckit as ofk
    >>> round(ofk.rastrigin([0, 0]), 4)
    0.0

    >>> round(ofk.rastrigin([1, 2]), 4)
    5.0

    >>> round(ofk.rastrigin([4.5, 4.5]), 4)
    80.5

    >>> round(ofk.rastrigin([1, 2, 3]), 4)
    14.0
    """
    x = check_vector(x)
    return float(10 * len(x) + (x**2 - 10 * np.cos(2 * np.pi * x)).sum())


def rosenbrock(x: Vector, /) -> float:
    """Rosenbrock function.

    A function :math:`f\\colon \\mathbb{R}^{n} \\rightarrow \\mathbb{R}`
    that takes an :math:`n`-vector (:math:`n > 1`) as input and returns a scalar value.
    [wiki]_

    .. math::

        f(\\mathbf{x}) =
        \\sum_{i=1}^{n-1} \\left(
            100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2
        \\right)

    Examples
    --------
    >>> import onekit.optfunckit as ofk
    >>> round(ofk.rosenbrock([0, 0]), 4)
    1.0

    >>> round(ofk.rosenbrock([1, 1]), 4)
    0.0

    >>> round(ofk.rosenbrock([1, 1, 1]), 4)
    0.0

    >>> round(ofk.rosenbrock([1, 2, 3]), 4)
    201.0

    >>> round(ofk.rosenbrock([3, 3]), 4)
    3604.0
    """
    x = check_vector(x, n_min=2)
    return float((100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).sum())


def schwefel(x: Vector, /) -> float:
    """Schwefel function.

    A function :math:`f\\colon \\mathbb{R}^{n} \\rightarrow \\mathbb{R}`
    that takes an :math:`n`-vector as input and returns a scalar value.
    [wiki]_ [schwefel]_

    .. math::

        f(\\mathbf{x}) =
        418.9829 n - \\sum_{i=1}^{n} x_{i} \\sin\\left( \\sqrt{|x_{i}|} \\right)

    References
    ----------
    .. [schwefel] "Schwefel function", Virtual Library of Simulation Experiments:
        Test Functions and Datasets, `<https://www.sfu.ca/~ssurjano/schwef.html>`_

    Examples
    --------
    >>> import onekit.optfunckit as ofk
    >>> round(ofk.schwefel([420.9687]), 4)
    0.0

    >>> round(ofk.schwefel([0, 0]), 4)
    837.9658

    >>> round(ofk.schwefel([1, 2]), 4)
    835.1488

    >>> round(ofk.schwefel([1, 2, 3]), 4)
    1251.1706
    """
    x = check_vector(x)
    n = len(x)
    return float(418.9829 * n - sum(x * np.sin(np.sqrt(np.abs(x)))))


def sinc(x: Vector, /) -> float:
    """Sinc function.

    A function :math:`f\\colon \\mathbb{R} \\rightarrow \\mathbb{R}`
    that takes an :math:`1`-vector as input and returns a scalar value.
    [sinc]_

    .. math::

        f(\\mathbf{x}) =
        \\begin{cases}
            \\frac{\\sin(x)}{x} & \\text{ if } x \\neq 0 \\\\
            1 & \\text{ if } x = 0
        \\end{cases}

    References
    ----------
    .. [sinc] "Sinc Function", Wolfram MathWorld,
        `<https://mathworld.wolfram.com/SincFunction.html>`_

    Examples
    --------
    >>> import onekit.optfunckit as ofk
    >>> ofk.sinc([0])
    1.0

    >>> round(ofk.sinc([1]), 4)
    0.8415
    """
    x = check_vector(x, n_min=1, n_max=1)[0]
    return float(1 if x == 0 else np.sin(x) / x)


def sphere(x: Vector, /) -> float:
    """Sphere function.

    A function :math:`f\\colon \\mathbb{R}^{n} \\rightarrow \\mathbb{R}`
    that takes an :math:`n`-vector as input and returns a scalar value.

    .. math::

       f(\\mathbf{x}) = \\sum_{i=1}^{n} x_i^2

    Examples
    --------
    >>> import onekit.optfunckit as ofk
    >>> ofk.sphere([0, 0])
    0.0

    >>> ofk.sphere([1, 1])
    2.0

    >>> ofk.sphere([1, 2, 3])
    14.0
    """
    x = check_vector(x)
    return float((x**2).sum())