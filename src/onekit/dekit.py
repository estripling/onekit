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
    def __init__(self, *individuals: Individual, key=None):
        super().__init__(check_individual_type(i) for i in pk.flatten(individuals))
        self._key = lambda ind: ind.fun if key is None else key

    @property
    def key(self) -> Callable:
        return self._key

    @property
    def size(self) -> int:
        return len(self)

    def evaluate(self, func: Callable[[Any], Any]) -> "Population":
        for individual in self:
            individual.evaluate(func)
        return self

    def sort(self, *, key=None, reverse=False) -> "Population":
        key = self.key if key is None else key
        self.data.sort(key=key, reverse=reverse)
        return self


def check_individual_type(individual: Individual) -> Individual:
    if not isinstance(individual, Individual):
        raise TypeError(f"{type(individual)=} - must be {Individual}")
    return individual
