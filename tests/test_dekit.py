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


@pytest.mark.parametrize("ind", [dek.Individual([0, 0]), None, 1, "two"])
def test_check_individual_type(ind: dek.Individual):
    if not isinstance(ind, dek.Individual):
        with pytest.raises(TypeError):
            dek.check_individual_type(ind)
