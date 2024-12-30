import numpy as np
import numpy.testing as npt
import pytest

import onekit.dekit as dek
import onekit.optfunckit as ofk
from onekit.dekit import (
    Bounds,
    Individual,
    Population,
)


class TestIndividual:
    def test_init(self):
        x = [1, 1]
        ind = dek.Individual(x)

        assert ind.x == x
        assert ind.fun is None
        assert ind.is_evaluated is False
        assert str(ind).startswith("None <-")

    # noinspection PyPropertyAccess
    def test_immutable_property(self):
        ind = dek.Individual([1, 1])

        with pytest.raises(AttributeError):
            ind.x = [2, 2]

        with pytest.raises(AttributeError):
            ind.fun = 3.0

    @pytest.mark.parametrize(
        "x, expected",
        [
            ([0], 0.0),
            ([0, 0], 0.0),
            ([1, 1], 3.6254),
            ([2, 2], 6.5936),
        ],
    )
    def test_evaluation(self, x: ofk.Vector, expected: float):
        ind = dek.Individual(x)

        assert np.all(ind.x == x)
        assert ind.fun is None
        assert ind.is_evaluated is False

        ind.evaluate(ofk.ackley)
        assert ind.fun is not None
        assert ind.is_evaluated is True

        assert round(ind.fun, 4) == expected


class TestPopulation:
    def test_init(self):
        ind1 = dek.Individual([0, 0])
        ind2 = dek.Individual([1, 1])
        ind3 = dek.Individual([2, 2])

        pop = dek.Population(ind1, ind2, ind3)
        assert pop == [ind1, ind2, ind3]

    def test_init__empty(self):
        pop = dek.Population()
        assert pop.size == 0
        assert not pop.is_evaluated

    @pytest.mark.parametrize("ind", [None, 1, "two"])
    def test_init__failed(self, ind: dek.Individual):
        with pytest.raises(TypeError):
            dek.Population(ind)

    def test_size(self):
        pop = dek.Population(dek.Individual(0), dek.Individual(1))
        assert pop.size == 2

    def test_list_methods(self):
        ind1 = dek.Individual([0, 0])
        ind2 = dek.Individual([1, 1])
        ind3 = dek.Individual([2, 2])

        pop = dek.Population(ind1, ind2, ind3)
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

        ind4 = dek.Individual([3, 3])
        ind5 = dek.Individual([4, 4])
        pop.extend([ind4, ind5])
        assert pop.size == 5
        assert pop == [ind1, ind2, ind3, ind4, ind5]

        pop.clear()
        assert pop.size == 0

        pop.extend([ind1, ind1])
        assert pop.size == 2
        assert pop == [ind1, ind1]

    def test_evaluate(self):
        ind1 = dek.Individual([0, 0])
        ind2 = dek.Individual([1, 1])
        ind3 = dek.Individual([2, 2])

        pop = dek.Population(ind1, ind2, ind3).evaluate(ofk.ackley)
        assert pop.is_evaluated

        expected = [0.0, 3.6254, 6.5936]
        assert all(round(ind.fun, 4) == fx for ind, fx in zip(pop, expected))

    def test_overwritten_sort(self):
        ind1 = dek.Individual([0, 0])
        ind2 = dek.Individual([1, 1])
        ind3 = dek.Individual([2, 2])

        pop = dek.Population(ind3, ind1, ind2).evaluate(ofk.ackley)
        assert pop.is_evaluated

        expected = [6.5936, 0.0, 3.6254]
        assert all(round(ind.fun, 4) == fx for ind, fx in zip(pop, expected))

        pop.sort()
        expected = [0.0, 3.6254, 6.5936]
        assert all(round(ind.fun, 4) == fx for ind, fx in zip(pop, expected))

        pop.sort(reverse=True)
        expected = [6.5936, 3.6254, 0.0]
        assert all(round(ind.fun, 4) == fx for ind, fx in zip(pop, expected))

    def test_min(self):
        ind1 = dek.Individual([0, 0])
        ind2 = dek.Individual([1, 1])
        ind3 = dek.Individual([2, 2])

        pop = dek.Population(ind3, ind1, ind2).evaluate(ofk.ackley)
        assert pop.is_evaluated

        actual = pop.min()
        expected = ind1
        assert actual == expected

    def test_max(self):
        ind1 = dek.Individual([0, 0])
        ind2 = dek.Individual([1, 1])
        ind3 = dek.Individual([2, 2])

        pop = dek.Population(ind3, ind1, ind2).evaluate(ofk.ackley)
        assert pop.is_evaluated

        actual = pop.max()
        expected = ind3
        assert actual == expected

    def test_incomplete_evaluation(self):
        ind1 = dek.Individual([0, 0])
        ind2 = dek.Individual([1, 1]).evaluate(ofk.ackley)
        ind3 = dek.Individual([2, 2]).evaluate(ofk.ackley)
        ind4 = dek.Individual([3, 3])

        pop = dek.Population(ind1, ind2, ind3, ind4).sort()
        assert not pop.is_evaluated
        assert pop == [ind1, ind4, ind2, ind3]
        assert pop.min() == ind1
        assert pop.max() == ind3


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
    def test_random__standard_uniform(self, seed: int):
        n_pop, n_dim = 10, 2
        init_strategy = dek.Initialization.random__standard_uniform(n_pop, n_dim, seed)
        pop = init_strategy()
        assert isinstance(pop, Population)
        assert pop.size == n_pop
        assert not pop.is_evaluated
        assert all(ind.x.dtype.kind in np.typecodes["AllFloat"] for ind in pop)
        assert all(ind.x.shape == (n_dim,) for ind in pop)
        mat = np.row_stack([ind.x for ind in pop])
        assert 0 <= mat.min() <= 1
        assert 0 <= mat.max() <= 1
        assert mat.min() < mat.max()

    def test_random__uniform(self, bounds: Bounds, seed: int):
        n_pop = 10
        init_strategy = dek.Initialization.random__uniform(n_pop, bounds, seed)
        pop = init_strategy()
        assert isinstance(pop, Population)
        assert pop.size == n_pop
        assert not pop.is_evaluated
        assert all(ind.x.dtype.kind in np.typecodes["AllFloat"] for ind in pop)
        assert all(ind.x.shape == (2,) for ind in pop)

    @pytest.fixture(scope="class")
    def bounds(self) -> Bounds:
        return [(-5, 5), (-5, 5)]

    @pytest.fixture(scope="class")
    def seed(self) -> int:
        return 101


class TestMutation:
    def test_rand_1(self, seed: int):
        n_pop, n_dim = 6, 2
        ind1 = dek.Individual(np.array([0, 0]))
        ind2 = dek.Individual(np.array([1, 1]))
        ind3 = dek.Individual(np.array([2, 2]))
        ind4 = dek.Individual(np.array([3, 3]))
        ind5 = dek.Individual(np.array([4, 4]))
        ind6 = dek.Individual(np.array([5, 5]))
        pop = dek.Population(ind1, ind2, ind3, ind4, ind5, ind6)

        mutation_strategy = dek.Mutation.rand_1(seed=seed)

        # target = ind1
        mutant1 = mutation_strategy(pop, ind1, 0.8)
        assert pop.size == n_pop
        assert not pop.is_evaluated
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
        assert not pop.is_evaluated
        assert pop == [ind1, ind2, ind3, ind4, ind5, ind6]
        assert isinstance(mutant2, Individual)
        assert mutant2 != ind2
        assert all(mutant2 != ind for ind in pop)
        assert not mutant2.is_evaluated
        assert mutant2.x.dtype.kind in np.typecodes["AllFloat"]
        assert mutant2.x.shape == (n_dim,)

        assert mutant1 != mutant2
        assert np.all(mutant1.x != mutant2.x)

    def test_best_1(self, seed: int):
        n_pop, n_dim = 6, 2
        ind1 = dek.Individual(np.array([0, 0]))
        ind2 = dek.Individual(np.array([1, 1]))
        ind3 = dek.Individual(np.array([2, 2]))
        ind4 = dek.Individual(np.array([3, 3]))
        ind5 = dek.Individual(np.array([4, 4]))
        ind6 = dek.Individual(np.array([5, 5]))
        pop = dek.Population(ind1, ind2, ind3, ind4, ind5, ind6).evaluate(ofk.sphere)
        assert pop.is_evaluated

        mutation_strategy = dek.Mutation.best_1(seed=seed)

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

    @pytest.fixture(scope="class")
    def seed(self) -> int:
        return 101


class TestBoundRepair:
    def test_identity(self):
        ind = dek.Individual(0)
        bound_repair_strategy = dek.BoundRepair.identity()
        actual = bound_repair_strategy(ind)
        assert actual == ind

    def test_clip__standard_uniform(self):
        ind1 = dek.Individual(np.array([0.1, 0.9]))
        ind2 = dek.Individual(np.array([0.1, 1.1]))
        ind3 = dek.Individual(np.array([-0.1, 0.9]))
        ind4 = dek.Individual(np.array([-0.1, 1.1]))

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
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_binomial_variant_1(self, seed: int):
        n_dim = 2
        target = dek.Individual(np.array([0, 0]))
        mutant = dek.Individual(np.array([1, 1]))

        crossover_strategy = dek.Crossover.binomial_variant_1(seed)

        trial = crossover_strategy(target, mutant, 0.7)
        assert isinstance(trial, Individual)
        assert trial != target
        assert trial != mutant
        assert trial.x.dtype.kind in np.typecodes["AllInteger"]
        assert trial.x.shape == (n_dim,)
        assert list(trial.x) in [[0, 1], [1, 0], [1, 1]]
        assert list(trial.x) != [0, 0]

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_binomial_variant_2(self, seed: int):
        n_dim = 2
        target = dek.Individual(np.array([0, 0]))
        mutant = dek.Individual(np.array([1, 1]))

        crossover_strategy = dek.Crossover.binomial_variant_2(seed)

        trial = crossover_strategy(target, mutant, 0.7)
        assert isinstance(trial, Individual)
        assert trial != target
        assert trial != mutant
        assert trial.x.dtype.kind in np.typecodes["AllInteger"]
        assert trial.x.shape == (n_dim,)
        assert list(trial.x) in [[0, 1], [1, 0]]
        assert list(trial.x) not in [[0, 0], [1, 1]]


class TestSelection:
    def test_smaller_function_value__target_selected(self):
        target = dek.Individual(np.array([0, 0])).evaluate(ofk.sphere)
        trail = dek.Individual(np.array([1, 1])).evaluate(ofk.sphere)
        selection_strategy = dek.Selection.smaller_function_value()

        winner = selection_strategy(target, trail)
        assert isinstance(winner, Individual)
        assert winner == target
        assert winner != trail

    def test_smaller_function_value__trail_selected(self):
        target = dek.Individual(np.array([1, 1])).evaluate(ofk.sphere)
        trail = dek.Individual(np.array([0, 0])).evaluate(ofk.sphere)
        selection_strategy = dek.Selection.smaller_function_value()

        winner = selection_strategy(target, trail)
        assert isinstance(winner, Individual)
        assert winner != target
        assert winner == trail

    def test_smaller_function_value__trail_selected__equal_values(self):
        target = dek.Individual(np.array([0, 0])).evaluate(ofk.sphere)
        trail = dek.Individual(np.array([0, 0])).evaluate(ofk.sphere)
        selection_strategy = dek.Selection.smaller_function_value()

        winner = selection_strategy(target, trail)
        assert isinstance(winner, Individual)
        assert winner != target
        assert winner == trail


@pytest.mark.parametrize("ind", [dek.Individual([0, 0]), None, 1, "two"])
def test_check_individual_type(ind: dek.Individual):
    if not isinstance(ind, dek.Individual):
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
