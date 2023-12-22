import numpy as np
import numpy.typing as npt

__all__ = (
    "check_vector",
    "stderr",
)


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


def stderr(x: ArrayLike, /) -> float:
    """Compute standard error of the mean.

    Examples
    --------
    >>> import numpy as np
    >>> import onekit.numpykit as npk
    >>> round(npk.stderr([98, 127, 82, 67, 121, np.nan, 119, 92, 110, 113, 107]), 4)
    5.9632
    """
    x = check_vector([v for v in x if np.isfinite(v)], n_min=0)
    n = len(x)
    return x.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
