import random

import numpy as np
import numpy.typing as npt

from onekit import mathkit as mk

__all__ = (
    "check_random_state",
    "check_vector",
    "create_boolean_array",
    "digitscale",
    "stderr",
)


ArrayLike = npt.ArrayLike
Seed = int | float | random.Random | np.random.RandomState | np.random.Generator | None
Vector = npt.NDArray[np.float64]


def check_random_state(seed: Seed = None) -> np.random.Generator:
    """Turn seed into np.random.Generator instance.

    Examples
    --------
    >>> import numpy as np
    >>> from onekit import numpykit as npk
    >>> rng = npk.check_random_state()
    >>> isinstance(rng, np.random.Generator)
    True
    """
    if isinstance(seed, np.random.Generator):
        return seed

    elif seed is None:
        return np.random.default_rng()

    elif isinstance(seed, int):
        return np.random.default_rng(seed)

    elif isinstance(seed, float):
        return np.random.default_rng(int(abs(seed)))

    elif isinstance(seed, random.Random):
        seed = seed.randint(1, np.iinfo(np.int32).max)
        return np.random.default_rng(seed)

    elif isinstance(seed, np.random.RandomState):
        seed = seed.random_integers(1, np.iinfo(np.int32).max, size=1)
        return np.random.default_rng(seed)

    else:
        raise ValueError(f"{seed=} - cannot be used to seed Generator instance")


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
    >>> from onekit import numpykit as npk
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


# noinspection PyTypeChecker
def create_boolean_array(data: ArrayLike, pos_label: int | str) -> np.ndarray:
    """Returns a boolean array indicating positions of pos_label in input data.

    Examples
    --------
    >>> from onekit import numpykit as npk
    >>> data = [0, 1, 2, 1, 0, 1]
    >>> npk.create_boolean_array(data, pos_label=1)
    array([False,  True, False,  True, False,  True])

    >>> data = ["cat", "dog", "cat", "bird", "cat", "dog"]
    >>> npk.create_boolean_array(data, pos_label="dog")
    array([False,  True, False, False, False,  True])
    """
    data_array = np.asarray(data)
    return data_array == pos_label


def digitscale(x: ArrayLike, /, *, kind: str = "log") -> np.ndarray:
    """NumPy version of digitscale.

    See Also
    --------
    onekit.mathkit.digitscale : Python version
    onekit.sparkkit.with_digitscale : PySpark version

    Examples
    --------
    >>> from onekit import numpykit as npk
    >>> npk.digitscale([0.1, 1, 10, 100, 1_000, 10_000, 2_000_000])
    array([0.     , 1.     , 2.     , 3.     , 4.     , 5.     , 7.30103])

    >>> npk.digitscale([0.1, 1, 10, 100, 1_000, 10_000, 100_000, 2_000_000], kind="int")
    array([0, 1, 2, 3, 4, 5, 6, 7])

    >>> npk.digitscale([0.2, 2, 20], kind="linear")
    array([0.11111111, 1.11111111, 2.11111111])
    """
    otypes = [int] if kind == "int" else [float]
    return np.vectorize(lambda x: mk.digitscale(x, kind=kind), otypes=otypes)(x)


def stderr(x: ArrayLike, /) -> float:
    """Compute standard error of the mean.

    Examples
    --------
    >>> import numpy as np
    >>> from onekit import numpykit as npk
    >>> np.set_printoptions(legacy="1.21")
    >>> round(npk.stderr([98, 127, 82, 67, 121, np.nan, 119, 92, 110, 113, 107]), 4)
    5.9632
    """
    x = check_vector([v for v in x if np.isfinite(v)], n_min=0)
    n = len(x)
    return x.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
