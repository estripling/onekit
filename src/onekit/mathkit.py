import math
from typing import Generator

import toolz

__all__ = (
    "collatz",
    "digitscale",
    "fibonacci",
    "isdivisible",
    "iseven",
    "isodd",
    "sign",
)


def collatz(n: int, /) -> Generator:
    """Generate a Collatz sequence.

    The famous 3n + 1 conjecture [c1]_ [c2]_. Given a positive integer :math:`n > 0`,
    the next term in the Collatz sequence is half of :math:`n`
    if :math:`n` is even; otherwise, if :math:`n` is odd,
    the next term is 3 times :math:`n` plus 1.
    Symbolically,

    .. math::

        f(n) =
        \\begin{cases}
             n / 2 & \\text{ if } n \\equiv 0 \\text{ (mod 2) } \\\\[6pt]
            3n + 1 & \\text{ if } n \\equiv 1 \\text{ (mod 2) }
        \\end{cases}

    The Collatz conjecture is that the sequence always reaches 1
    for any positive integer :math:`n`.

    Parameters
    ----------
    n : int
        A positive integer seeding the Collatz sequence.

    Yields
    ------
    int
        A generator of Collatz numbers that breaks when 1 is reached.

    Raises
    ------
    ValueError
        If ``n`` is not a positive integer.

    References
    ----------
    .. [c1] "Collatz", The On-Line Encyclopedia of Integer Sequences®,
            https://oeis.org/A006370
    .. [c2] "Collatz conjecture", Wikipedia,
            https://en.wikipedia.org/wiki/Collatz_conjecture

    Examples
    --------
    >>> import toolz
    >>> from onekit import mathkit as mk
    >>> n = 12
    >>> list(mk.collatz(n))
    [12, 6, 3, 10, 5, 16, 8, 4, 2, 1]
    >>> toolz.count(mk.collatz(n))
    10
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"{n=} - must be a positive integer")

    while True:
        yield n

        if n == 1:
            break

        # update
        n = n // 2 if iseven(n) else 3 * n + 1


def digitscale(x: int | float, /, *, kind: str = "log") -> int | float:
    """Scale :math:`x` such that its mapped integer part is its number of digits.

    Given a number :math:`x \\in \\mathbb{R}`, the following function
    :math:`f \\colon \\mathbb{R} \\rightarrow \\mathbb{R}_{\\ge 0}` scales it such that
    its mapped integer part :math:`\\lfloor f(x) \\rfloor \\in \\mathbb{N}_{0}`
    is the number of digits in :math:`[x]`:

    .. math::

        f(x) =
        \\begin{cases}
            1 + \\log_{10}|x| & \\text{ if } |x| \\ge 0.1 \\\\[6pt]
            0 & \\text{ otherwise }
        \\end{cases}

    Notes
    -----
    - :math:`\\lfloor \\cdot \\rfloor`: floor function
    - :math:`\\left[ \\, \\cdot \\, \\right]`: truncation function
    - For any positive integer :math:`k`, the number of digits in :math:`k` is
      :math:`1 + \\lfloor \\log_{10} k \\rfloor`
    - If `kind="int"`, returns :math:`\\lfloor f(x) \\rfloor`
    - If `kind="linear"`, linear interpolation is performed:

    .. math::

        f_{linear}(x) =
        \\begin{cases}
            \\frac{y_{0} (x_{1} - x) + y_{1} (x - x_{0})}{x_{1} - x_{0}}
              & \\text{ if } |x| \\ge 0.1 \\\\[6pt]
            0 & \\text{ otherwise }
        \\end{cases}

        \\\\[6pt]

        \\text{ with } n = \\lfloor f(x) \\rfloor, y_{0} = n, y_{1} = n + 1,
        x_{0} = 10^{n - 1}, \\text{ and } x_{1} = 10^{n}

    See Also
    --------
    onekit.numpykit.digitscale : NumPy version
    onekit.sparkkit.with_digitscale : PySpark version

    Examples
    --------
    >>> from onekit import mathkit as mk
    >>> list(map(mk.digitscale, [0.1, 1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]))
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

    >>> list(map(mk.digitscale, [0.2, 2, 20, 200]))
    [0.30102999566398125, 1.3010299956639813, 2.3010299956639813, 3.3010299956639813]

    >>> list(map(mk.digitscale, [-0.5, -5, -50, -500]))
    [0.6989700043360187, 1.6989700043360187, 2.6989700043360187, 3.6989700043360187]

    >>> list(map(lambda x: mk.digitscale(x, kind="int"), [-0.5, -5, -50, -500]))
    [0, 1, 2, 3]

    >>> list(map(lambda x: mk.digitscale(x, kind="linear"), [0.1, 1, 10, 100, 1_000]))
    [0.0, 1.0, 2.0, 3.0, 4.0]
    >>> list(map(lambda x: mk.digitscale(x, kind="linear"), [0.2, 2, 20, 200]))
    [0.11111111111111112, 1.1111111111111112, 2.111111111111111, 3.111111111111111]
    >>> list(map(lambda x: mk.digitscale(x, kind="linear"), [-0.5, -5, -50, -500]))
    [0.4444444444444445, 1.4444444444444444, 2.4444444444444446, 3.4444444444444446]
    """
    valid_kind = ["log", "int", "linear"]

    x = abs(x)
    fx = 1 + math.log10(x) if x >= 0.1 else 0.0

    if kind == "log":
        return fx

    elif kind == "int":
        return math.floor(fx)

    elif kind == "linear":
        n = math.floor(fx)
        y0, y1 = n, n + 1
        x0, x1 = 10 ** (n - 1), 10**n
        return (y0 * (x1 - x) + y1 * (x - x0)) / (x1 - x0) if x >= 0.1 else 0.0

    else:
        raise ValueError(f"{kind=} - must be a valid value: {valid_kind}")


def fibonacci() -> Generator:
    """Generate the Fibonacci sequence.

    For :math:`n > 1`, Fibonacci numbers may be defined by [f1]_ [f2]_:

    .. math::

        F(n) = F(n-1) + F(n-2) \\text{ with } F(0) = 0 \\text{ and } F(1) = 1.

    As such, the sequence starts as follows:

    .. math::

        0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, \\dots

    Yields
    ------
    int
        A generator of consecutive Fibonacci numbers.

    References
    ----------
    .. [f1] "Fibonacci numbers", The On-Line Encyclopedia of Integer Sequences®,
            https://oeis.org/A000045
    .. [f2] "Fibonacci number", Wikipedia,
            https://en.wikipedia.org/wiki/Fibonacci_number

    Examples
    --------
    >>> import toolz
    >>> from onekit import mathkit as mk
    >>> list(toolz.take(13, mk.fibonacci()))
    [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    """
    lag2, lag1 = 0, 1
    yield lag2
    yield lag1

    while True:
        lag0 = lag2 + lag1
        yield lag0
        lag2, lag1 = lag1, lag0


def isdivisible(x: int | float, /, n: int) -> bool:
    """Evaluate if :math:`x` is evenly divisible by :math:`n`.

    Examples
    --------
    >>> from functools import partial
    >>> from onekit import mathkit as mk
    >>> mk.isdivisible(49, 7)
    True

    >>> is_10_divisible_by = partial(mk.isdivisible, 10)
    >>> is_10_divisible_by(5)
    True
    >>> is_x_divisible_by_5 = partial(mk.isdivisible, n=5)
    >>> is_x_divisible_by_5(10)
    True
    >>> is_x_divisible_by_5(11.0)
    False
    """
    return x % n == 0


def iseven(x: int | float, /) -> bool:
    """Evaluate if :math:`x` is even.

    Examples
    --------
    >>> from onekit import mathkit as mk
    >>> mk.iseven(0)
    True

    >>> mk.iseven(1)
    False

    >>> mk.iseven(2)
    True
    """
    return isdivisible(x, n=2)


def isodd(x: int | float, /) -> bool:
    """Evaluate if :math:`x` is odd.

    Examples
    --------
    >>> from onekit import mathkit as mk
    >>> mk.isodd(0)
    False

    >>> mk.isodd(1)
    True

    >>> mk.isodd(2)
    False
    """
    return toolz.complement(iseven)(x)


def sign(x: int | float, /) -> int:
    """Sign function.

    .. math::

        f(x) =
        \\begin{cases}
            -1 & \\text{ if } x < 0 \\\\[6pt]
            0 & \\text{ if } x = 0 \\\\[6pt]
            1 & \\text{ if } x > 0
        \\end{cases}

    Examples
    --------
    >>> from onekit import mathkit as mk
    >>> mk.sign(0)
    0

    >>> mk.sign(3.14)
    1

    >>> mk.sign(-10)
    -1
    """
    return int(0 if math.isclose(x, 0) else math.copysign(1, x))
