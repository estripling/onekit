from typing import (
    Any,
    Callable,
)


class Individual:
    def __init__(self, x: Any):
        self._x = x
        self._fun = None

    @property
    def x(self) -> Any:
        return self._x

    @property
    def fun(self) -> Any | None:
        return self._fun

    @property
    def is_evaluated(self) -> bool:
        return False if self._fun is None else True

    def evaluate(self, func: Callable) -> None:
        if not self.is_evaluated:
            self._fun = func(self._x)
