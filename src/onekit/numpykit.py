import numpy as np
import numpy.typing as npt

__all__ = ("check_vector",)


ArrayLike = npt.ArrayLike
Vector = npt.NDArray[np.float64]


def check_vector(x: ArrayLike, /, *, n_min: int = 1, n_max: int = np.inf) -> Vector:
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
    >>> import onekit.numpykit as npk
    >>> npk.check_vector([0, 0])
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
