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
MutationStrategy = Callable[["Population", "Individual", float], "Individual"]
BoundRepairStrategy = Callable[["Individual"], "Individual"]
CrossoverStrategy = Callable[["Individual", "Individual", float], "Individual"]
SelectionStrategy = Callable[["Individual", "Individual"], "Individual"]


class Individual:
    def __init__(self, x: Any, /):
        self._x = x
        self._fun = None

    @property
    def x(self) -> Any:
        return self._x

    @property
    def fx(self) -> Any | None:
        return self._fun

    @property
    def is_evaluated(self) -> bool:
        return False if self._fun is None else True

    def evaluate(self, func: Callable[[Any], Any]) -> "Individual":
        if not self.is_evaluated:
            self._fun = func(self._x)
        return self

    def __repr__(self):
        return f"{self.fx} <- {self.x}"


class Population(UserList):
    def __init__(
        self,
        *individuals: Individual | Iterable[Individual],
        generation: int | None = None,
        key=None,
    ):
        super().__init__(check_individual_type(i) for i in pk.flatten(individuals))
        self._generation = generation
        self._key = (
            lambda ind: -float("inf")
            if ind.fx is None or not np.isfinite(ind.fx)
            else ind.fx
            if key is None
            else key
        )

    @property
    def generation(self) -> int | None:
        """Return the generation count of the population."""
        return self._generation

    @property
    def key(self) -> Callable:
        return self._key

    @property
    def size(self) -> int:
        return len(self)

    @property
    def is_evaluated(self) -> bool:
        return self.size > 0 and all(individual.is_evaluated for individual in self)

    def evaluate(self, func: Callable[[Any], Any]) -> "Population":
        for individual in self:
            individual.evaluate(func)
        return self

    def increment_generation_count(self) -> "Population":
        if self.generation is not None:
            self._generation += 1
        return self

    def copy(self) -> "Population":
        return Population(self.data, generation=self.generation, key=self.key)

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
    def x_bounds(self) -> np.ndarray:
        return np.array(self._bounds, dtype=np.int32).T

    @property
    def x_min(self) -> np.ndarray:
        return self.x_bounds[0, :]

    @property
    def x_max(self) -> np.ndarray:
        return self.x_bounds[1, :]

    @property
    def x_diff(self) -> np.ndarray:
        return self.x_max - self.x_min

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
            x_mat = rng.random((n_pop, n_dim))
            return Population((Individual(vec) for vec in x_mat), generation=0)

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
            x_mat = rng.random((n_pop, bnd.n_dim))
            return Population(
                (Individual(vec) for vec in bnd.x_min + bnd.x_diff * x_mat),
                generation=0,
            )

        return inner


class Mutation:
    @staticmethod
    def rand_1(seed: Seed) -> MutationStrategy:
        rng = npk.check_random_state(seed)

        def inner(
            population: Population,
            target: Individual,
            scale_factor: float,
            /,
        ) -> Individual:
            random_indices = rng.choice(
                [i for i in range(population.size) if i != population.index(target)],
                size=3,
                replace=False,
            )
            ind_r0, ind_r1, ind_r2 = (population[i] for i in random_indices)
            return Individual(ind_r0.x + scale_factor * (ind_r1.x - ind_r2.x))

        return inner

    @staticmethod
    def best_1(seed: Seed) -> MutationStrategy:
        rng = npk.check_random_state(seed)

        def inner(
            population: Population,
            target: Individual,
            scale_factor: float,
            /,
        ) -> Individual:
            ind_best = population.min()
            exclude_indices = {population.index(target), population.index(ind_best)}
            random_indices = rng.choice(
                [i for i in range(population.size) if i not in exclude_indices],
                size=2,
                replace=False,
            )
            ind_r1, ind_r2 = (population[i] for i in random_indices)
            return Individual(ind_best.x + scale_factor * (ind_r1.x - ind_r2.x))

        return inner


class BoundRepair:
    @staticmethod
    def identity() -> BoundRepairStrategy:
        def inner(ind: Individual, /) -> Individual:
            return ind

        return inner

    @staticmethod
    def clip__standard_uniform() -> BoundRepairStrategy:
        def inner(ind: Individual, /) -> Individual:
            if ((ind.x < 0) | (ind.x > 1)).any():
                ind = Individual(np.clip(ind.x, 0, 1))
            return ind

        return inner


class Crossover:
    @staticmethod
    def binomial_v1(seed: Seed) -> CrossoverStrategy:
        """Always for trail = mutant but never trail = target."""
        rng = npk.check_random_state(seed)

        def inner(target: Individual, mutant: Individual, prob: float, /) -> Individual:
            n_dim = len(target.x)
            xover_mask = rng.random(n_dim) <= prob

            if not xover_mask.any():
                j_rand = rng.integers(n_dim, size=1, dtype=np.uint32)
                xover_mask[j_rand] = True

            return Individual(np.where(xover_mask, mutant.x, target.x))

        return inner

    @staticmethod
    def binomial_v2(seed: Seed) -> CrossoverStrategy:
        """Makes sure trail != mutant and trail != target - always a mix."""
        rng = npk.check_random_state(seed)

        def inner(target: Individual, mutant: Individual, prob: float, /) -> Individual:
            n_dim = len(target.x)
            xover_mask = rng.random(n_dim) <= prob

            if not xover_mask.any():
                j_rand = rng.integers(n_dim, size=1, dtype=np.uint32)
                xover_mask[j_rand] = True

            if xover_mask.all():
                j_rand = rng.integers(n_dim, size=1, dtype=np.uint32)
                xover_mask[j_rand] = False

            return Individual(np.where(xover_mask, mutant.x, target.x))

        return inner


class Selection:
    @staticmethod
    def smaller_function_value() -> SelectionStrategy:
        def inner(
            target: Individual,
            trial: Individual,
            /,
        ) -> Individual:
            return trial if trial.fx <= target.fx else target

        return inner


def check_individual_type(individual: Individual) -> Individual:
    if not isinstance(individual, Individual):
        raise TypeError(f"{type(individual)=} - must be {Individual}")
    return individual


def check_bounds(bounds: Bounds) -> BoundsHandler:
    return BoundsHandler(bounds)


def denormalize(x: np.ndarray, x_min: np.ndarray, x_max: np.ndarray) -> np.ndarray:
    return x_min + (x_max - x_min) * x


def normalize(x: np.ndarray, x_min: np.ndarray, x_max: np.ndarray) -> np.ndarray:
    return (x - x_min) / (x_max - x_min)
