from typing import Generator

import onekit.pythonkit as pk

__all__ = (
    "collatz",
    "fibonacci",
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
    >>> import onekit.mathkit as mk
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
        n = n // 2 if pk.iseven(n) else 3 * n + 1


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
    >>> import onekit.mathkit as mk
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
