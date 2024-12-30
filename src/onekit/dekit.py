import random
from collections import UserList
from typing import (
    Any,
    Callable,
    Iterable,
    Sequence,
    Tuple,
)

import numpy as np

import onekit.numpykit as npk
import onekit.pythonkit as pk

Bounds = Sequence[Tuple[float, float]]
Seed = int | float | random.Random | np.random.RandomState | np.random.Generator | None
InitializationStrategy = Callable[[], "Population"]


class Individual:
    def __init__(self, x: Any, /):
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
    def __init__(self, *individuals: Individual | Iterable[Individual], key=None):
        super().__init__(check_individual_type(i) for i in pk.flatten(individuals))
        self._key = (
            lambda ind: -float("inf")
            if ind.fun is None or not np.isfinite(ind.fun)
            else ind.fun
            if key is None
            else key
        )

    @property
    def key(self) -> Callable:
        return self._key

    @property
    def size(self) -> int:
        return len(self)

    @property
    def is_evaluated(self) -> bool:
        return all(individual.is_evaluated for individual in self)

    def evaluate(self, func: Callable[[Any], Any]) -> "Population":
        for individual in self:
            individual.evaluate(func)
        return self

    def sort(self, *, key=None, reverse=False) -> "Population":
        key = self.key if key is None else key
        self.data.sort(key=key, reverse=reverse)
        return self

    def min(self, *, key=None) -> "Individual":
        key = self.key if key is None else key
        return min(self.data, key=key)

    def max(self, *, key=None) -> "Individual":
        key = self.key if key is None else key
        return max(self.data, key=key)


class BoundsHandler:
    def __init__(self, bounds: Bounds):
        self._bounds = bounds

    @property
    def bounds(self) -> Bounds:
        return self._bounds

    @property
    def lower_bounds(self) -> np.ndarray:
        return np.array([bound[0] for bound in self.bounds])

    @property
    def upper_bounds(self) -> np.ndarray:
        return np.array([bound[1] for bound in self.bounds])

    @property
    def scale(self) -> np.ndarray:
        return self.upper_bounds - self.lower_bounds

    @property
    def n_dim(self) -> int:
        return len(self._bounds)


class Initialization:
    @staticmethod
    def random__standard_uniform(
        n_pop: int,
        n_dim: int,
        random_state=Seed,
    ) -> InitializationStrategy:
        def inner():
            rng = npk.check_random_state(random_state)
            return Population(Individual(vec) for vec in rng.random((n_pop, n_dim)))

        return inner

    @staticmethod
    def random__uniform(
        n_pop: int,
        bounds: Bounds,
        random_state=Seed,
    ) -> InitializationStrategy:
        def inner():
            rng = npk.check_random_state(random_state)
            bnd = check_bounds(bounds)
            return Population(
                Individual(vec)
                for vec in bnd.lower_bounds + bnd.scale * rng.random((n_pop, bnd.n_dim))
            )

        return inner


def check_individual_type(individual: Individual) -> Individual:
    if not isinstance(individual, Individual):
        raise TypeError(f"{type(individual)=} - must be {Individual}")
    return individual


def check_bounds(bounds: Bounds) -> BoundsHandler:
    return BoundsHandler(bounds)
