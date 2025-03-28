from typing import Callable

import numpy as np
import numpy.testing as npt
import pytest

from onekit import optfunckit as ofk
from onekit import vizkit as vk


@pytest.mark.parametrize(
    "x, expected",
    [
        ([0], 0.0),
        ([0, 0], 0.0),
        ([1, 1], 3.6254),
        ([2, 2], 6.5936),
    ],
)
def test_ackley(x: ofk.Vector, expected: float):
    actual = ofk.ackley(x)
    assert round(actual, 4) == expected


@pytest.mark.parametrize(
    "x, expected",
    [
        ([3, 0.5], 0.0),
        ([0, 0], 14.2031),
        ([1, 1], 14.2031),
        ([2, 2], 356.7031),
    ],
)
def test_beale(x: ofk.Vector, expected: float):
    actual = ofk.beale(x)
    assert round(actual, 4) == expected


@pytest.mark.parametrize(
    "x, expected",
    [
        ([0], -0.3679),
        ([0, 0], -0.3679),
        ([0, 0, 0], -0.3679),
        ([0.5], -0.2636),
        ([0.5, 0.5], -0.1353),
        ([0.5, 0.5, 0.5], -0.0183),
        ([1], 0.0),
        ([1, 1], 0.0),
        ([1, 1, 1], 0.0),
    ],
)
def test_bump(x: ofk.Vector, expected: float):
    actual = ofk.bump(x)
    assert round(actual, 4) == expected


@pytest.mark.parametrize(
    "func, n, idx, expected_x, expected_fx",
    [
        (ofk.ackley, 1, 0, np.zeros(1), 0),
        (ofk.ackley, 2, 0, np.zeros(2), 0),
        (ofk.ackley, 5, 0, np.zeros(5), 0),
        (ofk.rastrigin, 5, 0, np.zeros(5), 0),
        (ofk.rosenbrock, 5, 0, np.ones(5), 0),
        (ofk.sinc, 1, 0, np.zeros(1), -1.0),
        (ofk.sphere, 5, 0, np.zeros(5), 0),
    ],
)
def test_fetch_minima(
    func: Callable,
    n: int,
    idx: int,
    expected_x: ofk.Vector,
    expected_fx: float,
):
    actual = ofk.fetch_minima(func, n)
    assert all(isinstance(opt, ofk.Minimum) for opt in actual)
    opt = actual[idx]
    npt.assert_array_almost_equal(opt.x, expected_x)
    assert opt.fx == expected_fx
    assert opt.n == n


def test_get_plotters__func1n():
    func1n_plotters = ofk.get_plotters__func1n()
    for name, plotter in func1n_plotters.items():
        assert isinstance(name, str)
        assert isinstance(plotter, vk.FunctionPlotter)
        assert isinstance(plotter.func, Callable)


def test_get_plotters__func2n():
    func2n_plotters = ofk.get_plotters__func2n()
    for name, plotter in func2n_plotters.items():
        assert isinstance(name, str)
        assert isinstance(plotter, vk.FunctionPlotter)
        assert isinstance(plotter.func, Callable)


@pytest.mark.parametrize(
    "fx, expected",
    [
        (-0.0, 0.0),
        (0.0, 0.0),
        (-1.0, 1.0),
        (1.0, -1.0),
    ],
)
def test_negate(fx: float, expected: float):
    actual = ofk.negate(fx)
    assert actual == expected


@pytest.mark.parametrize(
    "x, expected",
    [
        ([0, 0], 0.9810),
        ([1, 1], 2.4338),
        ([2, 2], 0.1328),
    ],
)
def test_peaks(x: ofk.Vector, expected: float):
    actual = ofk.peaks(x)
    assert round(actual, 4) == expected


@pytest.mark.parametrize(
    "x, expected",
    [
        ([0], 0.0),
        ([0, 0], 0.0),
        ([1, 1], 2.0),
        ([1, 2], 5.0),
        ([2, 1], 5.0),
        ([2, 2], 8.0),
        ([4, 4.5], 56.25),
        ([4.5, 4], 56.25),
        ([5.12, 5.12], 57.85),
    ],
)
def test_rastrigin(x: ofk.Vector, expected: float):
    actual = ofk.rastrigin(x)
    assert round(actual, 2) == expected


@pytest.mark.parametrize(
    "x, expected",
    [
        ([0, 0], 1.0),
        ([1, 1], 0.0),
        ([1, 1, 1], 0.0),
        ([1, 1, 1, 1], 0.0),
        ([1, 2], 100.0),
        ([2, 1], 901.0),
        ([2, 2], 401.0),
        ([4, 4.5], 13234.0),
        ([4.5, 4], 26418.5),
        ([5.12, 5.12], 44514.35),
    ],
)
def test_rosenbrock(x: ofk.Vector, expected: float):
    actual = ofk.rosenbrock(x)
    assert round(actual, 2) == expected


@pytest.mark.parametrize(
    "x, expected",
    [
        ([0], 418.9829),
        ([1], 418.1414),
        ([0, 0], 837.9658),
        ([1, 1], 836.2829),
        ([1, 2], 835.1488),
        ([1, 2, 3], 1251.1706),
        ([420.9687], 0),
        ([420.9687, 420.9687], 0),
        ([420.9687, 420.9687, 420.9687], 0),
    ],
)
def test_schwefel(x: ofk.Vector, expected: float):
    actual = ofk.schwefel(x)
    assert round(actual, 4) == expected


@pytest.mark.parametrize(
    "x, expected",
    [
        ([0], -1.0),
        ([1], -0.8415),
    ],
)
def test_sinc(x: ofk.Vector, expected: float):
    actual = ofk.sinc(x)
    assert round(actual, 4) == expected


@pytest.mark.parametrize(
    "x, expected",
    [
        ([0], 0),
        ([0, 0], 0),
        ([1, 1], 2),
        ([2, 2], 8),
    ],
)
def test_sphere(x: ofk.Vector, expected: float):
    actual = ofk.sphere(x)
    assert actual == expected
