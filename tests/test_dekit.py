import numpy as np
import pytest

import onekit.dekit as dek
import onekit.optfunckit as ofk


class TestIndividual:
    def test_init(self):
        x = [1, 1]
        ind = dek.Individual(x)

        assert ind.x == x
        assert ind.fun is None
        assert ind.is_evaluated is False

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


@pytest.mark.parametrize("ind", [dek.Individual([0, 0]), None, 1, "two"])
def test_check_individual_type(ind: dek.Individual):
    if not isinstance(ind, dek.Individual):
        with pytest.raises(TypeError):
            dek.check_individual_type(ind)
