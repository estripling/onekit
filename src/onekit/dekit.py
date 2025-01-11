import random
from abc import (
    ABC,
    abstractmethod,
)
from collections import UserList
from typing import (
    Any,
    Callable,
    Iterable,
    Sequence,
    Tuple,
)

import numpy as np
import toolz

import onekit.numpykit as npk
import onekit.pythonkit as pk

Bounds = Sequence[Tuple[float, float]]
Seed = int | float | random.Random | np.random.RandomState | np.random.Generator | None
ObjectiveFunction = Callable[[Any], Any]
InitializationStrategy = Callable[[], "Population"]
MutationStrategy = Callable[["Population", "Individual", float], "Individual"]
BoundRepairStrategy = Callable[["Individual"], "Individual"]
CrossoverStrategy = Callable[["Individual", "Individual", float], "Individual"]
ReplacementStrategy = Callable[["Individual", "Individual"], "Individual"]
TerminationStrategy = Callable[["DifferentialEvolution"], bool]
ParameterStrategy = Callable[[], float]


class Individual:
    def __init__(self, x: Any, /):
        self._x = x
        self._fx = None

    @property
    def x(self) -> Any:
        return self._x

    @property
    def fx(self) -> Any | None:
        return self._fx

    @fx.setter
    def fx(self, value: Any) -> None:
        if not self.is_evaluated:
            self._fx = value
        else:
            raise AttributeError("property 'fx' of 'Individual' object is already set")

    @property
    def is_evaluated(self) -> bool:
        return False if self._fx is None else True

    def __repr__(self):
        fx = f"{self._fx:g}" if isinstance(self._fx, float) else self._fx
        return f"<{self.__class__.__name__} {id(self)}> {self._x} {fx}"


class Population(UserList):
    def __init__(self, *individuals: Individual | Iterable[Individual], key=None):
        super().__init__(check_individual_type(i) for i in pk.flatten(individuals))
        self._key = KeyFunction.ind_fx() if key is None else key

    @property
    def key(self) -> Callable:
        return self._key

    @property
    def size(self) -> int:
        return len(self)

    @property
    def is_evaluated(self) -> bool:
        return self.size > 0 and all(individual.is_evaluated for individual in self)

    def copy(self) -> "Population":
        return Population(self.data, key=self.key)

    def shuffle(self, seed: Seed = None) -> "Population":
        rng = npk.check_random_state(seed)
        rng.shuffle(self)
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


class KeyFunction:
    @staticmethod
    def ind_fx():
        return lambda ind: ind.fx

    @staticmethod
    def neg_inf():
        return (
            lambda ind: -float("inf")
            if ind.fx is None or not np.isfinite(ind.fx)
            else ind.fx
        )


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
    def identity(population: Population) -> InitializationStrategy:
        def inner() -> Population:
            return population

        return inner

    @staticmethod
    def random__standard_uniform(
        n_pop: int,
        n_dim: int,
        random_state=Seed,
    ) -> InitializationStrategy:
        def inner() -> Population:
            rng = npk.check_random_state(random_state)
            x_mat = rng.random((n_pop, n_dim))
            return Population((Individual(vec) for vec in x_mat))

        return inner

    @staticmethod
    def random__uniform(
        n_pop: int,
        bounds: Bounds,
        random_state=Seed,
    ) -> InitializationStrategy:
        def inner() -> Population:
            rng = npk.check_random_state(random_state)
            bnd = check_bounds(bounds)
            x_mat = rng.random((n_pop, bnd.n_dim))
            return Population(Individual(vec) for vec in bnd.x_min + bnd.x_diff * x_mat)

        return inner


class Mutation:
    @staticmethod
    def rand_1(seed: Seed) -> MutationStrategy:
        """rand/1"""
        rng = npk.check_random_state(seed)

        def inner(
            population: Population,
            target: Individual,
            f: float,
            /,
        ) -> Individual:
            exclude = population.index(target)
            indices = tuple(i for i in range(population.size) if i != exclude)
            r0, r1, r2 = (
                population[i] for i in rng.choice(indices, size=3, replace=False)
            )
            return Individual(r0.x + f * (r1.x - r2.x))

        return inner

    @staticmethod
    def best_1(seed: Seed) -> MutationStrategy:
        """best/1"""
        rng = npk.check_random_state(seed)

        def inner(
            population: Population,
            target: Individual,
            f: float,
            /,
        ) -> Individual:
            best = population.min()
            exclude = {population.index(target), population.index(best)}
            indices = tuple(i for i in range(population.size) if i not in exclude)
            r1, r2 = (population[i] for i in rng.choice(indices, size=2, replace=False))
            return Individual(best.x + f * (r1.x - r2.x))

        return inner

    @staticmethod
    def rand_to_best(seed: Seed) -> MutationStrategy:
        """rand-to-best/1"""
        rng = npk.check_random_state(seed)

        def inner(
            population: Population,
            target: Individual,
            f: float,
            /,
        ) -> Individual:
            best = population.min()
            exclude = {population.index(target), population.index(best)}
            indices = tuple(i for i in range(population.size) if i not in exclude)
            r0, r1, r2 = (
                population[i] for i in rng.choice(indices, size=3, replace=False)
            )
            return Individual(r0.x + f * (best.x - target.x) + f * (r1.x - r2.x))

        return inner

    @staticmethod
    def current_to_best(seed: Seed) -> MutationStrategy:
        """current-to-best/1"""
        rng = npk.check_random_state(seed)

        def inner(
            population: Population,
            target: Individual,
            f: float,
            /,
        ) -> Individual:
            best = population.min()
            exclude = {population.index(target), population.index(best)}
            indices = tuple(i for i in range(population.size) if i not in exclude)
            r1, r2 = (population[i] for i in rng.choice(indices, size=2, replace=False))
            return Individual(target.x + f * (best.x - target.x) + f * (r1.x - r2.x))

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
    def binomial(seed: Seed) -> CrossoverStrategy:
        """Trail is either a crossover between mutant and target or just the mutant."""
        rng = npk.check_random_state(seed)

        def inner(target: Individual, mutant: Individual, cr: float, /) -> Individual:
            n_dim = len(target.x)
            xover_mask = rng.random(n_dim) < cr

            if not xover_mask.any():
                j_rand = rng.integers(n_dim, size=1, dtype=np.uint32)
                xover_mask[j_rand] = True

            return Individual(np.where(xover_mask, mutant.x, target.x))

        return inner

    @staticmethod
    def binomial_v2(seed: Seed) -> CrossoverStrategy:
        """Always makes sure that trail is a crossover between mutant and target."""
        rng = npk.check_random_state(seed)

        def inner(target: Individual, mutant: Individual, cr: float, /) -> Individual:
            n_dim = len(target.x)
            xover_mask = rng.random(n_dim) < cr

            if not xover_mask.any():
                j_rand = rng.integers(n_dim, size=1, dtype=np.uint32)
                xover_mask[j_rand] = True

            if xover_mask.all():
                j_rand = rng.integers(n_dim, size=1, dtype=np.uint32)
                xover_mask[j_rand] = False

            return Individual(np.where(xover_mask, mutant.x, target.x))

        return inner


class Replacement:
    @staticmethod
    def smaller_is_better() -> ReplacementStrategy:
        def inner(target: Individual, trial: Individual, /) -> Individual:
            return trial if trial.fx <= target.fx else target

        return inner


class Termination:
    @staticmethod
    def has_reached_max_generations(max_generations: int, /) -> TerminationStrategy:
        def inner(de: DifferentialEvolution, /) -> bool:
            if de.generation_count >= max_generations:
                de.message = "reached max generations"
                return True
            return False

        return inner

    @staticmethod
    def has_reached_max_evaluations(max_evaluations: int, /) -> TerminationStrategy:
        def inner(de: DifferentialEvolution, /) -> bool:
            if de.evaluation_count >= max_evaluations:
                de.message = "reached max evaluations"
                return True
            return False

        return inner

    @staticmethod
    def has_reached_max_best_so_far(max_best_so_far: int, /) -> TerminationStrategy:
        def inner(de: DifferentialEvolution, /) -> bool:
            if de.best_so_far_count >= max_best_so_far:
                de.message = "reached max best-so-far"
                return True
            return False

        return inner

    @staticmethod
    def have_fx_values_converged(
        abs_tol: float = 0.0,
        rel_tol: float = 0.01,
    ) -> TerminationStrategy:
        def inner(de: DifferentialEvolution, /) -> bool:
            fxs = np.array([ind.fx for ind in de.population])
            if fxs.std() <= abs_tol + rel_tol * np.abs(fxs.mean()):
                de.message = "fx values converged"
                return True
            return False

        return inner

    @staticmethod
    def has_met_any_strategy(
        *termination_strategy: TerminationStrategy | Iterable[TerminationStrategy],
    ) -> TerminationStrategy:
        def inner(de: DifferentialEvolution, /) -> bool:
            return pk.are_predicates_true(any, termination_strategy)(de)

        return inner

    @staticmethod
    def has_met_any_basic_strategy(
        max_generations: int | None = None,
        max_evaluations: int | None = None,
        max_best_so_far: int | None = None,
        abs_tol: float = 0.0,
        rel_tol: float = 0.01,
    ) -> TerminationStrategy:
        max_generations = float("inf") if max_generations is None else max_generations
        max_evaluations = float("inf") if max_evaluations is None else max_evaluations
        max_best_so_far = float("inf") if max_best_so_far is None else max_best_so_far
        termination_strategy = Termination.has_met_any_strategy(
            Termination.has_reached_max_generations(max_generations),
            Termination.has_reached_max_evaluations(max_evaluations),
            Termination.has_reached_max_best_so_far(max_best_so_far),
            Termination.have_fx_values_converged(abs_tol, rel_tol),
        )

        def inner(de: DifferentialEvolution, /) -> bool:
            return termination_strategy(de)

        return inner


class Parameter:
    @staticmethod
    def constant(value: float) -> ParameterStrategy:
        def inner() -> float:
            return value

        return inner

    @staticmethod
    def dither(low: float, high: float, seed: Seed) -> ParameterStrategy:
        rng = npk.check_random_state(seed)

        def inner() -> float:
            return float(rng.uniform(low, high, size=1))

        return inner


def evaluate_individual(func: ObjectiveFunction, individual: Individual) -> Individual:
    if not individual.is_evaluated:
        individual.fx = func(individual.x)
    return individual


def evaluate_population(func: ObjectiveFunction, population: Population) -> Population:
    for i, individual in enumerate(population):
        population[i] = evaluate_individual(func, individual)
    return population


def evaluate(func: ObjectiveFunction, obj: Individual | Population) -> int:
    """Evaluate individual or population object and return evaluation count.

    Notes
    -----
    This is not a pure function.
    If the 'Individual' object is not evaluated, the function return value is assigned
    to its property 'fx' as a side effect.

    Returns
    -------
    int
        Evaluation count of non-evaluated individuals.
    """
    if isinstance(obj, Individual):
        if not obj.is_evaluated:
            obj.fx = func(obj.x)
            return 1
        return 0

    elif isinstance(obj, Population):
        return sum(evaluate(func, individual) for individual in obj)

    else:
        raise TypeError(f"{type(obj)=} - must be {Individual} or {Population}")


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


class DifferentialEvolution(ABC):
    def __init__(
        self,
        func: ObjectiveFunction,
        init_strategy: InitializationStrategy,
        mutation_strategy: MutationStrategy,
        bound_repair_strategy: BoundRepairStrategy,
        crossover_strategy: CrossoverStrategy,
        replacement_strategy: ReplacementStrategy,
        termination_strategy: TerminationStrategy,
        f_strategy: ParameterStrategy,
        cr_strategy: ParameterStrategy,
    ):
        self.func = func
        self.init_strategy = init_strategy
        self.mutation_strategy = mutation_strategy
        self.bound_repair_strategy = bound_repair_strategy
        self.crossover_strategy = crossover_strategy
        self.replacement_strategy = replacement_strategy
        self.termination_strategy = termination_strategy
        self.f_strategy = f_strategy
        self.cr_strategy = cr_strategy
        self.population: Population | None = None
        self.generation_count: int = 0
        self.evaluation_count: int = 0
        self.best_so_far_count: int = 0
        self.best_so_far_value: float = float("inf")
        self.message: str | None = None

    def __iter__(self) -> "DifferentialEvolution":
        return self

    @abstractmethod
    def __next__(self) -> "DifferentialEvolution":
        pass  # pragma: no cover

    @property
    def best(self) -> Individual:
        """Returns the best solution in the current population."""
        return self.population.min()

    @property
    def worst(self) -> Individual:
        """Returns the worst solution in the current population."""
        return self.population.max()

    def _init_population(self) -> "DifferentialEvolution":
        self.population = self.init_strategy()
        self.evaluation_count += evaluate(self.func, self.population)
        return self

    def _update_best_so_far(self) -> "DifferentialEvolution":
        fx_best = self.best.fx
        if fx_best < self.best_so_far_value:
            self.best_so_far_value = fx_best
            self.best_so_far_count = 1
        else:
            self.best_so_far_count += 1
        return self


class DeV1(DifferentialEvolution):
    """Differential Evolution Variant 1: Classic DE."""

    def __next__(self) -> "DifferentialEvolution":
        if self.population is None:
            return self._init_population()

        if self.termination_strategy(self):
            raise StopIteration

        new_population = Population()
        f = self.f_strategy()
        cr = self.cr_strategy()

        for target in self.population:
            trial = toolz.pipe(
                self.mutation_strategy(self.population, target, f),
                lambda mutant: self.bound_repair_strategy(mutant),
                lambda mutant: self.crossover_strategy(target, mutant, cr),
            )
            self.evaluation_count += evaluate(self.func, trial)
            survivor = self.replacement_strategy(target, trial)
            new_population.append(survivor)

        self.population = new_population
        self.generation_count += 1
        self._update_best_so_far()

        return self


class DeV2(DifferentialEvolution):
    """Differential Evolution Variant 2:
    - A small deviation from the classic DE.
    - That is, fitter solutions are immediately updated within a single generation.
    """

    def __next__(self) -> "DifferentialEvolution":
        if self.population is None:
            return self._init_population()

        if self.termination_strategy(self):
            raise StopIteration

        f = self.f_strategy()
        cr = self.cr_strategy()

        for i, target in enumerate(self.population):
            trial = toolz.pipe(
                self.mutation_strategy(self.population, target, f),
                lambda mutant: self.bound_repair_strategy(mutant),
                lambda mutant: self.crossover_strategy(target, mutant, cr),
            )
            self.evaluation_count += evaluate(self.func, trial)
            survivor = self.replacement_strategy(target, trial)
            self.population[i] = survivor

        self.generation_count += 1
        self._update_best_so_far()

        return self
