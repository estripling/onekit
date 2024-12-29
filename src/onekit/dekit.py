from collections import UserList
from typing import (
    Any,
    Callable,
)

import onekit.pythonkit as pk


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

    def evaluate(self, func: Callable[[Any], Any]) -> "Individual":
        if not self.is_evaluated:
            self._fun = func(self._x)
        return self

    def __repr__(self):
        return f"{self.fun} <- {self.x}"


class Population(UserList):
    def __init__(self, *individuals: Individual):
        super().__init__(check_individual_type(i) for i in pk.flatten(individuals))

    @property
    def size(self) -> int:
        return len(self)


def check_individual_type(individual: Individual) -> Individual:
    if not isinstance(individual, Individual):
        raise TypeError(f"{type(individual)=} - must be {Individual}")
    return individual
