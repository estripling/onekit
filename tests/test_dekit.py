import numpy as np
import numpy.testing as npt
import pytest
import toolz

import onekit.dekit as dek
import onekit.numpykit as npk
import onekit.optfunckit as ofk
from onekit.dekit import (
    Bounds,
    DifferentialEvolution,
    Individual,
    ObjectiveFunction,
    Population,
)


class TestIndividual:
    def test_init(self):
        x = [1, 1]
        ind = Individual(x)

        assert ind.x == x
        assert ind.fx is None
        assert ind.is_evaluated is False
        assert str(ind).startswith(f"<{Individual.__name__}")

    # noinspection PyPropertyAccess
    def test_immutable_property__x(self):
        ind = Individual([1, 1])

        with pytest.raises(AttributeError):
            ind.x = [2, 2]

    # noinspection PyPropertyAccess
    def test_immutable_property__fx(self):
        ind = Individual([1, 1])
        ind.fx = 3.0

        with pytest.raises(AttributeError):
            ind.fx = 3.0


@pytest.mark.parametrize(
    "x, expected",
    [
        ([0], 0.0),
        ([0, 0], 0.0),
        ([1, 1], 3.6254),
        ([2, 2], 6.5936),
    ],
)
def test_evaluate_individual(x: ofk.Vector, expected: float):
    ind = Individual(x)

    assert np.all(ind.x == x)
    assert ind.fx is None
    assert ind.is_evaluated is False

    ind = dek.evaluate_individual(ofk.ackley, ind)
    assert ind.fx is not None
    assert ind.is_evaluated is True

    assert round(ind.fx, 4) == expected


class TestPopulation:
    def test_init(self):
        ind1 = Individual([0, 0])
        ind2 = Individual([1, 1])
        ind3 = Individual([2, 2])

        pop = Population(ind1, ind2, ind3)
        assert pop == [ind1, ind2, ind3]

    def test_init__empty(self):
        pop = Population()
        assert pop.size == 0
        assert not pop.is_evaluated

    def test_init__empty__generation_count(self):
        pop = Population()
        assert pop.size == 0
        assert not pop.is_evaluated

    @pytest.mark.parametrize("ind", [None, 1, "two"])
    def test_init__failed(self, ind: Individual):
        with pytest.raises(TypeError):
            Population(ind)

    def test_size(self):
        pop = Population(Individual(0), Individual(1))
        assert pop.size == 2

    def test_list_methods(self):
        ind1 = Individual([0, 0])
        ind2 = Individual([1, 1])
        ind3 = Individual([2, 2])

        pop = Population(ind1, ind2, ind3)
        assert pop.size == 3
        assert pop.index(ind1) == 0
        assert pop.count(ind1) == 1
        assert pop[0] == ind1
        assert pop[-1] == ind3
        assert pop[:-1] == [ind1, ind2]

        ind = pop.pop(0)
        assert pop.size == 2
        assert ind == ind1
        assert pop == [ind2, ind3]

        pop.append(ind)
        assert pop.size == 3
        assert pop == [ind2, ind3, ind1]

        pop.remove(ind1)
        assert pop.size == 2
        assert pop == [ind2, ind3]

        pop.insert(0, ind1)
        assert pop.size == 3
        assert pop == [ind1, ind2, ind3]

        pop.reverse()
        assert pop.size == 3
        assert pop == [ind3, ind2, ind1]
        pop.reverse()
        assert pop == [ind1, ind2, ind3]

        ind4 = Individual([3, 3])
        ind5 = Individual([4, 4])
        pop.extend([ind4, ind5])
        assert pop.size == 5
        assert pop == [ind1, ind2, ind3, ind4, ind5]

        pop.clear()
        assert pop.size == 0

        pop.extend([ind1, ind1])
        assert pop.size == 2
        assert pop == [ind1, ind1]

    def test_evaluate(self):
        ind1 = Individual([0, 0])
        ind2 = Individual([1, 1])
        ind3 = Individual([2, 2])

        pop = dek.evaluate_population(ofk.ackley, Population(ind1, ind2, ind3))
        assert pop.is_evaluated

        expected = [0.0, 3.6254, 6.5936]
        assert all(round(ind.fx, 4) == fx for ind, fx in zip(pop, expected))

    def test_overwritten_copy(self):
        ind1 = Individual([0, 0])
        ind2 = Individual([1, 1])
        ind3 = Individual([2, 2])

        pop1 = Population(ind1, ind2)
        assert pop1 == [ind1, ind2]
        assert pop1.size == 2
        assert not pop1.is_evaluated

        pop2 = pop1.copy()
        assert isinstance(pop2, Population)
        assert pop2 == [ind1, ind2]
        assert pop2.size == 2
        assert not pop2.is_evaluated

        pop2.append(ind3)
        assert pop1 == [ind1, ind2]
        assert pop1.size == 2
        assert not pop2.is_evaluated

        assert pop2 == [ind1, ind2, ind3]
        assert pop2.size == 3
        assert not pop2.is_evaluated

    def test_overwritten_sort(self):
        ind1 = Individual([0, 0])
        ind2 = Individual([1, 1])
        ind3 = Individual([2, 2])

        pop = dek.evaluate_population(ofk.ackley, Population(ind3, ind1, ind2))
        assert pop.is_evaluated

        expected = [6.5936, 0.0, 3.6254]
        assert all(round(ind.fx, 4) == fx for ind, fx in zip(pop, expected))

        pop.sort()
        expected = [0.0, 3.6254, 6.5936]
        assert all(round(ind.fx, 4) == fx for ind, fx in zip(pop, expected))

        pop.sort(reverse=True)
        expected = [6.5936, 3.6254, 0.0]
        assert all(round(ind.fx, 4) == fx for ind, fx in zip(pop, expected))

    @pytest.mark.parametrize("sample_size", [1, 2, 3])
    def test_sample(self, sample_size: int):
        ind1 = Individual(1)
        ind2 = Individual(2)
        ind3 = Individual(3)
        ind4 = Individual(4)
        individuals = [ind1, ind2, ind3, ind4]
        pop = Population(individuals)

        assert pop.size == len(individuals)
        assert pop == individuals

        actual = pop.sample(size=sample_size, exclude=[ind1], seed=101)
        assert pop.size == len(individuals)
        assert pop == individuals
        assert isinstance(actual, Population)
        assert actual.size == sample_size
        assert ind1 not in actual
        assert sum(ind in individuals for ind in actual) == sample_size
        assert actual != [ind2, ind3, ind4]  # random order

    def test_shuffle(self):
        individuals = [Individual(1), Individual(2), Individual(3), Individual(4)]
        pop = Population(individuals)

        assert pop.size == len(individuals)
        assert pop == individuals

        pop.shuffle(101)
        assert pop.size == len(individuals)
        assert all(ind in individuals for ind in pop)
        assert pop != individuals

    def test_min(self):
        ind1 = Individual([0, 0])
        ind2 = Individual([1, 1])
        ind3 = Individual([2, 2])

        pop = dek.evaluate_population(ofk.ackley, Population(ind3, ind1, ind2))
        assert pop.is_evaluated

        actual = pop.min()
        expected = ind1
        assert actual == expected

    def test_max(self):
        ind1 = Individual([0, 0])
        ind2 = Individual([1, 1])
        ind3 = Individual([2, 2])

        pop = dek.evaluate_population(ofk.ackley, Population(ind3, ind1, ind2))
        assert pop.is_evaluated

        actual = pop.max()
        expected = ind3
        assert actual == expected

    def test_incomplete_evaluation(self):
        ind1 = Individual([0, 0])
        ind2 = dek.evaluate_individual(ofk.ackley, Individual([1, 1]))
        ind3 = dek.evaluate_individual(ofk.ackley, Individual([2, 2]))
        ind4 = Individual([3, 3])

        pop = Population(ind1, ind2, ind3, ind4, key=dek.KeyFunction.neg_inf()).sort()
        assert not pop.is_evaluated
        assert pop == [ind1, ind4, ind2, ind3]
        assert pop.min() == ind1
        assert pop.max() == ind3

    # noinspection PyPropertyAccess
    def test_immutable_property(self):
        pop = Population(Individual([0, 0]))

        with pytest.raises(AttributeError):
            pop.key = None

        with pytest.raises(AttributeError):
            pop.size = 100

        with pytest.raises(AttributeError):
            pop.is_evaluated = None


class TestBoundsHandler:
    def test_init(
        self,
        bounds: Bounds,
        x_bounds: np.ndarray,
        x_min: np.ndarray,
        x_max: np.ndarray,
        x_diff: np.ndarray,
    ):
        actual = dek.check_bounds(bounds)
        assert actual.bounds == bounds
        npt.assert_array_equal(actual.x_bounds, x_bounds)
        npt.assert_array_equal(actual.x_min, x_min)
        npt.assert_array_equal(actual.x_max, x_max)
        npt.assert_array_equal(actual.x_diff, x_diff)
        assert actual.n_dim == 2

    @pytest.fixture(scope="class")
    def bounds(self) -> Bounds:
        return [(-5, 5)] * 2

    @pytest.fixture(scope="class")
    def x_bounds(self) -> Bounds:
        return np.array([(-5, -5), (5, 5)])

    @pytest.fixture(scope="class")
    def x_min(self) -> np.ndarray:
        return np.array([-5] * 2)

    @pytest.fixture(scope="class")
    def x_max(self) -> np.ndarray:
        return np.array([5] * 2)

    @pytest.fixture(scope="class")
    def x_diff(self) -> np.ndarray:
        return np.array([10] * 2)


class TestInitialization:
    def test_identity(self):
        pop = Population(
            Individual(np.array([0, 0])),
            Individual(np.array([1, 1])),
            Individual(np.array([2, 2])),
            Individual(np.array([3, 3])),
        )
        init_strategy = dek.Initialization.identity(pop)
        actual = init_strategy()
        assert actual == pop

    def test_random__standard_uniform(self, seed: int):
        n_pop, n_dim = 10, 2
        init_strategy = dek.Initialization.random__standard_uniform(n_pop, n_dim, seed)
        actual = init_strategy()
        assert isinstance(actual, Population)
        assert actual.size == n_pop
        assert not actual.is_evaluated
        assert all(ind.x.dtype.kind in np.typecodes["AllFloat"] for ind in actual)
        assert all(ind.x.shape == (n_dim,) for ind in actual)
        mat = np.row_stack([ind.x for ind in actual])
        assert 0 <= mat.min() <= 1
        assert 0 <= mat.max() <= 1
        assert mat.min() < mat.max()

    def test_random__uniform(self, bounds: Bounds, seed: int):
        n_pop = 10
        init_strategy = dek.Initialization.random__uniform(n_pop, bounds, seed)
        actual = init_strategy()
        assert isinstance(actual, Population)
        assert actual.size == n_pop
        assert not actual.is_evaluated
        assert all(ind.x.dtype.kind in np.typecodes["AllFloat"] for ind in actual)
        assert all(ind.x.shape == (2,) for ind in actual)

    @pytest.fixture(scope="class")
    def bounds(self) -> Bounds:
        return [(-5, 5), (-5, 5)]

    @pytest.fixture(scope="class")
    def seed(self) -> int:
        return 101


class TestMutation:
    @pytest.mark.parametrize(
        "func",
        [
            dek.Mutation.rand_1,
            dek.Mutation.best_1,
            dek.Mutation.rand_to_best_1,
            dek.Mutation.current_to_best_1,
        ],
    )
    def test_mutation_strategies(self, func, seed: int):
        mutation_strategy = func(seed)

        n_pop, n_dim = 6, 2
        ind1 = Individual(np.array([0, 0]))
        ind2 = Individual(np.array([1, 1]))
        ind3 = Individual(np.array([2, 2]))
        ind4 = Individual(np.array([3, 3]))
        ind5 = Individual(np.array([4, 4]))
        ind6 = Individual(np.array([5, 5]))
        pop = dek.evaluate_population(
            ofk.sphere,
            Population(ind1, ind2, ind3, ind4, ind5, ind6),
        )
        assert pop.is_evaluated

        # target = ind1
        mutant1 = mutation_strategy(pop, ind1, 0.8)
        assert pop.size == n_pop
        assert pop == [ind1, ind2, ind3, ind4, ind5, ind6]
        assert isinstance(mutant1, Individual)
        assert mutant1 != ind1
        assert all(mutant1 != ind for ind in pop)
        assert not mutant1.is_evaluated
        assert mutant1.x.dtype.kind in np.typecodes["AllFloat"]
        assert mutant1.x.shape == (n_dim,)

        # target = ind2
        mutant2 = mutation_strategy(pop, ind2, 0.8)
        assert pop.size == n_pop
        assert pop == [ind1, ind2, ind3, ind4, ind5, ind6]
        assert isinstance(mutant2, Individual)
        assert mutant2 != ind2
        assert all(mutant2 != ind for ind in pop)
        assert not mutant2.is_evaluated
        assert mutant2.x.dtype.kind in np.typecodes["AllFloat"]
        assert mutant2.x.shape == (n_dim,)

        assert mutant1 != mutant2
        assert np.all(mutant1.x != mutant2.x)

    @pytest.mark.parametrize(
        "func",
        [
            dek.Mutation.rand_to_pbest_1,
            dek.Mutation.current_to_pbest_1,
        ],
    )
    def test_pbest_mutation_strategies(self, func, seed: int):
        mutation_strategy = func(seed)

        n_pop, n_dim = 4, 2
        ind1 = Individual(np.array([0, 0]))
        ind2 = Individual(np.array([1, 1]))
        ind3 = Individual(np.array([2, 2]))
        ind4 = Individual(np.array([3, 3]))
        ind5 = Individual(np.array([4, 4]))
        ind6 = Individual(np.array([5, 5]))
        pop = dek.evaluate_population(ofk.sphere, Population(ind1, ind2, ind3, ind4))
        archive = dek.evaluate_population(ofk.sphere, Population(ind5, ind6))
        assert pop.is_evaluated
        assert archive.is_evaluated

        f = 0.8
        p = 0.2
        mutant1 = mutation_strategy(pop, ind1, f, p, None)
        assert isinstance(mutant1, Individual)
        assert pop.size == n_pop
        assert pop == [ind1, ind2, ind3, ind4]
        assert isinstance(mutant1, Individual)
        assert mutant1 != ind1
        assert all(mutant1 != ind for ind in pop)
        assert not mutant1.is_evaluated
        assert mutant1.x.dtype.kind in np.typecodes["AllFloat"]
        assert mutant1.x.shape == (n_dim,)

        mutant2 = mutation_strategy(pop, ind2, f, p, archive)
        assert isinstance(mutant2, Individual)
        assert pop.size == n_pop
        assert pop == [ind1, ind2, ind3, ind4]
        assert isinstance(mutant2, Individual)
        assert mutant2 != ind1
        assert all(mutant2 != ind for ind in pop)
        assert not mutant2.is_evaluated
        assert mutant2.x.dtype.kind in np.typecodes["AllFloat"]
        assert mutant2.x.shape == (n_dim,)

    @pytest.fixture(scope="class")
    def seed(self) -> int:
        return 101


class TestBoundRepair:
    def test_identity(self):
        ind = Individual(0)
        bound_repair_strategy = dek.BoundRepair.identity()
        actual = bound_repair_strategy(ind)
        assert actual == ind

    def test_clip__standard_uniform(self):
        ind1 = Individual(np.array([0.1, 0.9]))
        ind2 = Individual(np.array([0.1, 1.1]))
        ind3 = Individual(np.array([-0.1, 0.9]))
        ind4 = Individual(np.array([-0.1, 1.1]))

        bound_repair_strategy = dek.BoundRepair.clip__standard_uniform()

        actual = bound_repair_strategy(ind1)
        assert actual == ind1

        actual = bound_repair_strategy(ind2)
        assert actual != ind2
        npt.assert_array_equal(actual.x, np.array([0.1, 1.0]))

        actual = bound_repair_strategy(ind3)
        assert actual != ind3
        npt.assert_array_equal(actual.x, np.array([0.0, 0.9]))

        actual = bound_repair_strategy(ind4)
        assert actual != ind4
        npt.assert_array_equal(actual.x, np.array([0.0, 1.0]))


class TestCrossover:
    @pytest.mark.parametrize("cr", [0.0, 0.7, 1.0])
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_binomial(self, cr: float, seed: int):
        n_dim = 2
        target = Individual(np.array([0, 0]))
        mutant = Individual(np.array([1, 1]))

        crossover_strategy = dek.Crossover.binomial(seed)

        trial = crossover_strategy(target, mutant, cr)
        assert isinstance(trial, Individual)
        assert trial != target
        assert trial != mutant
        assert trial.x.dtype.kind in np.typecodes["AllInteger"]
        assert trial.x.shape == (n_dim,)
        assert list(trial.x) in [[0, 1], [1, 0], [1, 1]]
        assert list(trial.x) != [0, 0]

    @pytest.mark.parametrize("cr", [0.0, 0.7, 1.0])
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_binomial_v2(self, cr: float, seed: int):
        n_dim = 2
        target = Individual(np.array([0, 0]))
        mutant = Individual(np.array([1, 1]))

        crossover_strategy = dek.Crossover.binomial_v2(seed)

        trial = crossover_strategy(target, mutant, cr)
        assert isinstance(trial, Individual)
        assert trial != target
        assert trial != mutant
        assert trial.x.dtype.kind in np.typecodes["AllInteger"]
        assert trial.x.shape == (n_dim,)
        assert list(trial.x) in [[0, 1], [1, 0]]
        assert list(trial.x) not in [[0, 0], [1, 1]]


class TestReplacement:
    def test_smaller_function_value__target_selected(self):
        target = dek.evaluate_individual(ofk.sphere, Individual(np.array([0, 0])))
        trail = dek.evaluate_individual(ofk.sphere, Individual(np.array([1, 1])))
        replacement_strategy = dek.Replacement.smaller_is_better()

        actual = replacement_strategy(target, trail)
        assert isinstance(actual, Individual)
        assert actual == target
        assert actual != trail

    def test_smaller_function_value__trail_selected(self):
        target = dek.evaluate_individual(ofk.sphere, Individual(np.array([1, 1])))
        trail = dek.evaluate_individual(ofk.sphere, Individual(np.array([0, 0])))
        replacement_strategy = dek.Replacement.smaller_is_better()

        actual = replacement_strategy(target, trail)
        assert isinstance(actual, Individual)
        assert actual != target
        assert actual == trail

    def test_smaller_function_value__trail_selected__equal_values(self):
        target = dek.evaluate_individual(ofk.sphere, Individual(np.array([0, 0])))
        trail = dek.evaluate_individual(ofk.sphere, Individual(np.array([0, 0])))
        replacement_strategy = dek.Replacement.smaller_is_better()

        actual = replacement_strategy(target, trail)
        assert isinstance(actual, Individual)
        assert actual != target
        assert actual == trail


class TestTermination:
    def test_has_reached_max_generations(self, seed: int):
        termination_strategy = dek.Termination.has_reached_max_generations(1)

        rng = npk.check_random_state(seed)
        pop = Population(
            Individual(np.array([0, 0])),
            Individual(np.array([1, 1])),
            Individual(np.array([2, 2])),
            Individual(np.array([3, 3])),
        )
        de = dek.DeV1(
            func=ofk.sphere,
            init_strategy=dek.Initialization.identity(pop),
            mutation_strategy=dek.Mutation.rand_1(rng),
            bound_repair_strategy=dek.BoundRepair.identity(),
            crossover_strategy=dek.Crossover.binomial(rng),
            replacement_strategy=dek.Replacement.smaller_is_better(),
            termination_strategy=termination_strategy,
            f_strategy=dek.Parameter.constant(0.8),
            cr_strategy=dek.Parameter.constant(0.9),
        )
        assert de.evaluation_count == 0

        next(de)
        assert de.population is pop
        assert de.population == pop
        assert de.population.size == pop.size
        assert de.generation_count == 0
        assert de.evaluation_count == 4
        assert de.message is None

        actual = termination_strategy(de)
        assert isinstance(actual, bool)
        assert actual is False
        assert de.population is pop
        assert de.population == pop
        assert de.population.size == pop.size
        assert de.generation_count == 0
        assert de.evaluation_count == 4
        assert de.message is None

        next(de)
        actual = termination_strategy(de)
        assert isinstance(actual, bool)
        assert actual is True
        assert de.population is not pop
        assert de.population != pop
        assert de.population.size == pop.size
        assert de.generation_count == 1
        assert de.evaluation_count == 8
        assert de.message == "reached max generations"

    def test_has_reached_max_evaluations(self, seed: int):
        termination_strategy = dek.Termination.has_reached_max_evaluations(8)

        rng = npk.check_random_state(seed)
        pop = Population(
            Individual(np.array([0, 0])),
            Individual(np.array([1, 1])),
            Individual(np.array([2, 2])),
            Individual(np.array([3, 3])),
        )
        de = dek.DeV1(
            func=ofk.sphere,
            init_strategy=dek.Initialization.identity(pop),
            mutation_strategy=dek.Mutation.rand_1(rng),
            bound_repair_strategy=dek.BoundRepair.identity(),
            crossover_strategy=dek.Crossover.binomial(rng),
            replacement_strategy=dek.Replacement.smaller_is_better(),
            termination_strategy=termination_strategy,
            f_strategy=dek.Parameter.constant(0.8),
            cr_strategy=dek.Parameter.constant(0.9),
        )
        assert de.evaluation_count == 0

        next(de)
        assert de.population is pop
        assert de.population == pop
        assert de.population.size == pop.size
        assert de.generation_count == 0
        assert de.evaluation_count == 4
        assert de.message is None

        actual = termination_strategy(de)
        assert isinstance(actual, bool)
        assert actual is False
        assert de.population is pop
        assert de.population == pop
        assert de.population.size == pop.size
        assert de.generation_count == 0
        assert de.evaluation_count == 4
        assert de.message is None

        next(de)
        actual = termination_strategy(de)
        assert isinstance(actual, bool)
        assert actual is True
        assert de.population is not pop
        assert de.population != pop
        assert de.population.size == pop.size
        assert de.generation_count == 1
        assert de.evaluation_count == 8
        assert de.message == "reached max evaluations"

    def test_has_reached_max_best_so_far(self, seed: int):
        termination_strategy = dek.Termination.has_reached_max_best_so_far(1)

        rng = npk.check_random_state(seed)
        pop = Population(
            Individual(np.array([0, 0])),
            Individual(np.array([1, 1])),
            Individual(np.array([2, 2])),
            Individual(np.array([3, 3])),
        )
        de = dek.DeV1(
            func=ofk.sphere,
            init_strategy=dek.Initialization.identity(pop),
            mutation_strategy=dek.Mutation.rand_1(rng),
            bound_repair_strategy=dek.BoundRepair.identity(),
            crossover_strategy=dek.Crossover.binomial(rng),
            replacement_strategy=dek.Replacement.smaller_is_better(),
            termination_strategy=termination_strategy,
            f_strategy=dek.Parameter.constant(0.8),
            cr_strategy=dek.Parameter.constant(0.9),
        )
        assert de.evaluation_count == 0
        assert de.best_so_far_count == 0

        next(de)
        assert de.population is pop
        assert de.population == pop
        assert de.population.size == pop.size
        assert de.generation_count == 0
        assert de.evaluation_count == 4
        assert de.best_so_far_count == 0
        assert np.isinf(de.best_so_far_value)
        assert de.message is None

        actual = termination_strategy(de)
        assert isinstance(actual, bool)
        assert actual is False
        assert de.population is pop
        assert de.population == pop
        assert de.population.size == pop.size
        assert de.generation_count == 0
        assert de.evaluation_count == 4
        assert de.best_so_far_count == 0
        assert np.isinf(de.best_so_far_value)
        assert de.message is None

        next(de)
        actual = termination_strategy(de)
        assert isinstance(actual, bool)
        assert actual is True
        assert de.population is not pop
        assert de.population != pop
        assert de.population.size == pop.size
        assert de.generation_count == 1
        assert de.evaluation_count == 8
        assert de.best_so_far_count == 1
        assert de.best_so_far_value == 0
        assert de.message == "reached max best-so-far"

    def test_have_fx_values_converged(self, seed: int):
        termination_strategy = dek.Termination.have_fx_values_converged(rel_tol=0.9)

        rng = npk.check_random_state(seed)
        pop = Population(
            Individual(np.array([0, 0])),
            Individual(np.array([1, 1])),
            Individual(np.array([2, 2])),
            Individual(np.array([3, 3])),
        )
        de = dek.DeV1(
            func=ofk.sphere,
            init_strategy=dek.Initialization.identity(pop),
            mutation_strategy=dek.Mutation.rand_1(rng),
            bound_repair_strategy=dek.BoundRepair.identity(),
            crossover_strategy=dek.Crossover.binomial(rng),
            replacement_strategy=dek.Replacement.smaller_is_better(),
            termination_strategy=termination_strategy,
            f_strategy=dek.Parameter.constant(0.8),
            cr_strategy=dek.Parameter.constant(0.7),
        )
        assert de.evaluation_count == 0

        next(de)
        assert de.population is pop
        assert de.population == pop
        assert de.population.size == pop.size
        assert de.generation_count == 0
        assert de.evaluation_count == 4
        assert de.message is None

        actual = termination_strategy(de)
        assert isinstance(actual, bool)
        assert actual is False
        assert de.population is pop
        assert de.population == pop
        assert de.population.size == pop.size
        assert de.generation_count == 0
        assert de.evaluation_count == 4
        assert de.message is None

        next(de)
        actual = termination_strategy(de)
        assert isinstance(actual, bool)
        assert actual is True
        assert de.population is not pop
        assert de.population != pop
        assert de.population.size == pop.size
        assert de.generation_count == 1
        assert de.evaluation_count == 8
        assert de.message == "fx values converged"

    def test_has_met_any_basic_strategy__max_generations(self, seed: int):
        termination_strategy = dek.Termination.has_met_any_basic_strategy(
            max_generations=1
        )

        rng = npk.check_random_state(seed)
        pop = Population(
            Individual(np.array([0, 0])),
            Individual(np.array([1, 1])),
            Individual(np.array([2, 2])),
            Individual(np.array([3, 3])),
        )
        de = dek.DeV1(
            func=ofk.sphere,
            init_strategy=dek.Initialization.identity(pop),
            mutation_strategy=dek.Mutation.rand_1(rng),
            bound_repair_strategy=dek.BoundRepair.identity(),
            crossover_strategy=dek.Crossover.binomial(rng),
            replacement_strategy=dek.Replacement.smaller_is_better(),
            termination_strategy=termination_strategy,
            f_strategy=dek.Parameter.constant(0.8),
            cr_strategy=dek.Parameter.constant(0.9),
        )
        assert de.evaluation_count == 0

        next(de)
        assert de.population is pop
        assert de.population == pop
        assert de.population.size == pop.size
        assert de.generation_count == 0
        assert de.evaluation_count == 4

        actual = termination_strategy(de)
        assert isinstance(actual, bool)
        assert actual is False
        assert de.population is pop
        assert de.population == pop
        assert de.population.size == pop.size
        assert de.generation_count == 0
        assert de.evaluation_count == 4

        next(de)
        actual = termination_strategy(de)
        assert isinstance(actual, bool)
        assert actual is True
        assert de.population is not pop
        assert de.population != pop
        assert de.population.size == pop.size
        assert de.generation_count == 1
        assert de.evaluation_count == 8
        assert de.message == "reached max generations"

    def test_has_met_any_basic_strategy__max_evaluations(self, seed: int):
        termination_strategy = dek.Termination.has_met_any_basic_strategy(
            max_evaluations=8
        )

        rng = npk.check_random_state(seed)
        pop = Population(
            Individual(np.array([0, 0])),
            Individual(np.array([1, 1])),
            Individual(np.array([2, 2])),
            Individual(np.array([3, 3])),
        )
        de = dek.DeV1(
            func=ofk.sphere,
            init_strategy=dek.Initialization.identity(pop),
            mutation_strategy=dek.Mutation.rand_1(rng),
            bound_repair_strategy=dek.BoundRepair.identity(),
            crossover_strategy=dek.Crossover.binomial(rng),
            replacement_strategy=dek.Replacement.smaller_is_better(),
            termination_strategy=termination_strategy,
            f_strategy=dek.Parameter.constant(0.8),
            cr_strategy=dek.Parameter.constant(0.9),
        )
        assert de.evaluation_count == 0

        next(de)
        assert de.population is pop
        assert de.population == pop
        assert de.population.size == pop.size
        assert de.generation_count == 0
        assert de.evaluation_count == 4

        actual = termination_strategy(de)
        assert isinstance(actual, bool)
        assert actual is False
        assert de.population is pop
        assert de.population == pop
        assert de.population.size == pop.size
        assert de.generation_count == 0
        assert de.evaluation_count == 4

        next(de)
        actual = termination_strategy(de)
        assert isinstance(actual, bool)
        assert actual is True
        assert de.population is not pop
        assert de.population != pop
        assert de.population.size == pop.size
        assert de.generation_count == 1
        assert de.evaluation_count == 8
        assert de.message == "reached max evaluations"

    def test_has_met_any_basic_strategy__fx_values_converged(self, seed: int):
        termination_strategy = dek.Termination.has_met_any_basic_strategy(rel_tol=0.9)

        rng = npk.check_random_state(seed)
        pop = Population(
            Individual(np.array([0, 0])),
            Individual(np.array([1, 1])),
            Individual(np.array([2, 2])),
            Individual(np.array([3, 3])),
        )
        de = dek.DeV1(
            func=ofk.sphere,
            init_strategy=dek.Initialization.identity(pop),
            mutation_strategy=dek.Mutation.rand_1(rng),
            bound_repair_strategy=dek.BoundRepair.identity(),
            crossover_strategy=dek.Crossover.binomial(rng),
            replacement_strategy=dek.Replacement.smaller_is_better(),
            termination_strategy=termination_strategy,
            f_strategy=dek.Parameter.constant(0.8),
            cr_strategy=dek.Parameter.constant(0.7),
        )
        assert de.evaluation_count == 0

        next(de)
        assert de.population is pop
        assert de.population == pop
        assert de.population.size == pop.size
        assert de.generation_count == 0
        assert de.evaluation_count == 4

        actual = termination_strategy(de)
        assert isinstance(actual, bool)
        assert actual is False
        assert de.population is pop
        assert de.population == pop
        assert de.population.size == pop.size
        assert de.generation_count == 0
        assert de.evaluation_count == 4

        next(de)
        actual = termination_strategy(de)
        assert isinstance(actual, bool)
        assert actual is True
        assert de.population is not pop
        assert de.population != pop
        assert de.population.size == pop.size
        assert de.generation_count == 1
        assert de.evaluation_count == 8
        assert de.message == "fx values converged"

    @pytest.fixture(scope="class")
    def seed(self) -> int:
        return 101


class TestParameter:
    def test_constant(self):
        parameter_strategy = dek.Parameter.constant(0.9)
        actual = parameter_strategy()
        assert actual == 0.9

    def test_dither(self):
        parameter_strategy = dek.Parameter.dither(0.5, 1.0, seed=101)
        actual = parameter_strategy()
        assert 0.5 <= actual < 1.0


class TestEvaluation:
    def test_evaluate_individual(self, func: ObjectiveFunction):
        ind = Individual([1, 1])
        assert not ind.is_evaluated

        ind = dek.evaluate_individual(func, ind)
        assert ind.fx == 2

    def test_evaluate_population(self, func):
        pop = Population(Individual([1, 1]), Individual([2, 2]))
        assert not pop.is_evaluated

        pop = dek.evaluate_population(func, pop)
        assert sum(ind.fx for ind in pop) == 10

    def test_evaluate__individual(self, func: ObjectiveFunction):
        ind = Individual([1, 1])
        assert not ind.is_evaluated

        count = dek.evaluate(func, ind)
        assert ind.fx == 2
        assert count == 1

        count = dek.evaluate(func, ind)
        assert ind.fx == 2
        assert count == 0

    def test_evaluate__population(self, func: ObjectiveFunction):
        pop = Population(Individual([1, 1]), Individual([2, 2]))
        assert not pop.is_evaluated

        count = dek.evaluate(func, pop)
        assert sum(ind.fx for ind in pop) == 10
        assert count == 2

        count = dek.evaluate(func, pop)
        assert sum(ind.fx for ind in pop) == 10
        assert count == 0

        pop.append(Individual([3, 3]))
        count = dek.evaluate(func, pop)
        assert sum(ind.fx for ind in pop) == 28
        assert count == 1

    # noinspection PyTypeChecker
    @pytest.mark.parametrize("obj", [None, 0, 1.0, "two"])
    def test_evaluate__invalid_input(self, func: ObjectiveFunction, obj: object):
        with pytest.raises(TypeError):
            dek.evaluate(func, obj)

    @pytest.fixture(scope="class")
    def func(self) -> ObjectiveFunction:
        return ofk.sphere


@pytest.mark.parametrize("ind", [Individual([0, 0]), None, 1, "two"])
def test_check_individual_type(ind: Individual):
    if not isinstance(ind, Individual):
        with pytest.raises(TypeError):
            dek.check_individual_type(ind)


@pytest.mark.parametrize(
    "x, expected",
    [
        (np.array([0.0, 0.0]), np.array([-5.0, -5.0])),
        (np.array([0.5, 0.5]), np.array([0.0, 0.0])),
        (np.array([1.0, 1.0]), np.array([5.0, 5.0])),
    ],
)
def test_denormalize(x: np.ndarray, expected: np.ndarray):
    actual = dek.denormalize(x, x_min=np.array([-5] * 2), x_max=np.array([5] * 2))
    npt.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "x, expected",
    [
        (np.array([-5.0, -5.0]), np.array([0.0, 0.0])),
        (np.array([0.0, 0.0]), np.array([0.5, 0.5])),
        (np.array([5.0, 5.0]), np.array([1.0, 1.0])),
    ],
)
def test_normalize(x: np.ndarray, expected: np.ndarray):
    actual = dek.normalize(x, x_min=np.array([-5] * 2), x_max=np.array([5] * 2))
    npt.assert_array_equal(actual, expected)


class TestDifferentialEvolution:
    @pytest.mark.parametrize("cls", [dek.DeV1, dek.DeV2])
    def test_classic_de(self, cls, func: ObjectiveFunction, bounds: Bounds, seed: int):
        rng = npk.check_random_state(seed)
        bnh = dek.check_bounds(bounds)

        def denorm(x: np.ndarray) -> np.ndarray:
            return dek.denormalize(x, x_min=bnh.x_min, x_max=bnh.x_max)

        def problem(x: np.ndarray) -> ObjectiveFunction:
            return toolz.pipe(x, denorm, func)

        d = bnh.n_dim
        n = 10 * d

        de = cls(
            func=problem,
            init_strategy=dek.Initialization.random__standard_uniform(n, d, rng),
            mutation_strategy=dek.Mutation.rand_1(rng),
            bound_repair_strategy=dek.BoundRepair.clip__standard_uniform(),
            crossover_strategy=dek.Crossover.binomial(rng),
            replacement_strategy=dek.Replacement.smaller_is_better(),
            termination_strategy=dek.Termination.has_met_any_basic_strategy(
                max_generations=20
            ),
            f_strategy=dek.Parameter.dither(0.5, 1.0, rng),
            cr_strategy=dek.Parameter.constant(0.9),
        )

        for g, generation in enumerate(de):
            assert isinstance(generation, DifferentialEvolution)
            assert generation.generation_count == g
            assert generation.evaluation_count == (g + 1) * n

            assert isinstance(generation.population, Population)
            assert generation.population.size == n

            assert generation.best.fx < generation.worst.fx

        solution = de.best
        assert isinstance(solution, Individual)
        x_best = denorm(solution.x)
        fx_best = solution.fx
        npt.assert_array_almost_equal(x_best, np.zeros(d), decimal=2)
        npt.assert_almost_equal(fx_best, 0, decimal=2)

    @pytest.mark.parametrize("cls", [dek.DeV3, dek.DeV4])
    def test_shade(self, cls, func: ObjectiveFunction, bounds: Bounds, seed: int):
        rng = npk.check_random_state(seed)
        bnh = dek.check_bounds(bounds)

        def denorm(x: np.ndarray) -> np.ndarray:
            return dek.denormalize(x, x_min=bnh.x_min, x_max=bnh.x_max)

        def problem(x: np.ndarray) -> ObjectiveFunction:
            return toolz.pipe(x, denorm, func)

        d = bnh.n_dim
        n = 10 * d

        de = cls(
            func=problem,
            init_strategy=dek.Initialization.random__standard_uniform(n, d, rng),
            mutation_strategy=dek.Mutation.current_to_pbest_1(rng),
            bound_repair_strategy=dek.BoundRepair.clip__standard_uniform(),
            crossover_strategy=dek.Crossover.binomial(rng),
            replacement_strategy=dek.Replacement.smaller_is_better(),
            termination_strategy=dek.Termination.has_met_any_basic_strategy(
                max_generations=20
            ),
            memory_size=5,
            seed=rng,
        )

        for g, generation in enumerate(de):
            assert isinstance(generation, DifferentialEvolution)
            assert generation.generation_count == g
            assert generation.evaluation_count == (g + 1) * n

            assert isinstance(generation.population, Population)
            assert generation.population.size == n

            assert generation.best.fx < generation.worst.fx

        solution = de.best
        assert isinstance(solution, Individual)
        x_best = denorm(solution.x)
        fx_best = solution.fx
        npt.assert_array_almost_equal(x_best, np.zeros(d), decimal=2)
        npt.assert_almost_equal(fx_best, 0, decimal=2)

    @pytest.fixture(scope="class")
    def func(self) -> ObjectiveFunction:
        return ofk.sphere

    @pytest.fixture(scope="class")
    def bounds(self) -> Bounds:
        return [(-2, 2)] * 2

    @pytest.fixture(scope="class")
    def seed(self) -> int:
        return 101
