__all__ = ("num_to_str",)


def num_to_str(n: int or float, /) -> str:
    """Format a number to string with underscores as thousands separator.

    Examples
    --------
    >>> import onekit
    >>> onekit.num_to_str(1000000)
    '1_000_000'

    >>> onekit.num_to_str(100000.0)
    '100_000.0'
    """
    return f"{n:_}"
