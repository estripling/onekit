from typing import (
    Callable,
    List,
)

import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits
import numpy as np
import numpy.testing as npt
import pytest
import toolz

import onekit.optfunckit as ofk
import onekit.vizkit as vk


class TestFunctionPlotter:
    def test_init(self, func: Callable, bounds1n: List[vk.Pair]):
        actual = vk.FunctionPlotter(func, bounds1n)
        assert isinstance(actual.func, Callable)
        assert actual.bounds == bounds1n

    def test_init_with_invalid_bounds(self, func: Callable):
        with pytest.raises(TypeError):
            vk.FunctionPlotter(func, bounds=[])

        with pytest.raises(TypeError):
            vk.FunctionPlotter(func, bounds=[(-5, 5)] * 3)

    def test_init_with_invalid_bool_args(self, func: Callable, bounds2n: List[vk.Pair]):
        with pytest.raises(ValueError):
            vk.FunctionPlotter(func, bounds2n, with_surface=False, with_contour=False)

    def test_default_plot__1d(self, func: Callable, bounds1n: List[vk.Pair]):
        plotter = vk.FunctionPlotter(func, bounds1n)
        fig, ax, ax3d = plotter.plot()
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
        assert ax3d is None

    def test_default_plot__2d(self, func: Callable, bounds2n: List[vk.Pair]):
        plotter = vk.FunctionPlotter(func, bounds2n)
        fig, ax, ax3d = plotter.plot()
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
        assert isinstance(ax3d, mpl_toolkits.mplot3d.axes3d.Axes3D)

    def test_surface_plot(self, func: Callable, bounds2n: List[vk.Pair]):
        plotter = vk.FunctionPlotter(func, bounds2n, with_contour=False)
        fig, ax, ax3d = plotter.plot()
        assert isinstance(fig, matplotlib.figure.Figure)
        assert ax is None
        assert isinstance(ax3d, mpl_toolkits.mplot3d.axes3d.Axes3D)

    def test_contour_plot(self, func: Callable, bounds2n: List[vk.Pair]):
        plotter = vk.FunctionPlotter(func, bounds2n, with_surface=False)
        fig, ax, ax3d = plotter.plot()
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
        assert ax3d is None

    def test_plot_with_points(self, func: Callable, bounds2n: List[vk.Pair]):
        plotter = vk.FunctionPlotter(func, bounds2n, points=[vk.Point(0, 0, 0)])
        fig, ax, ax3d = plotter.plot()
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
        assert isinstance(ax3d, mpl_toolkits.mplot3d.axes3d.Axes3D)

    def test_default_plot__ackley(self, bounds2n: List[vk.Pair]):
        plotter = vk.FunctionPlotter(ofk.ackley, bounds2n)
        fig, ax, ax3d = plotter.plot()
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
        assert isinstance(ax3d, mpl_toolkits.mplot3d.axes3d.Axes3D)

    @pytest.fixture
    def func(self) -> Callable:
        return lambda x: (x**2).sum()

    @pytest.fixture
    def bounds1n(self) -> List[vk.Pair]:
        return [(-5.0, 5.0)]

    @pytest.fixture
    def bounds2n(self) -> List[vk.Pair]:
        return [(-5.0, 5.0)] * 2


@pytest.mark.parametrize(
    "method_name",
    [
        "get_kws_contour__base",
        "get_kws_contourf__base",
        "get_kws_contourf__YlOrBr",
        "get_kws_contourf__YlOrBr_r",
        "get_kws_plot__base",
        "get_kws_scatter__base",
        "get_kws_surface__base",
        "get_kws_surface__YlOrBr",
        "get_kws_surface__YlOrBr_r",
    ],
)
def test_config(method_name: str):
    actual = getattr(vk.Config, method_name)()
    assert isinstance(actual, dict)


def test_create_xy_points():
    actual = vk.create_xy_points(ofk.sphere, [-2, -1, 0, 1, 2])
    expected = vk.XyPoints(
        x=np.array([-2, -1, 0, 1, 2]),
        y=np.array([4.0, 1.0, 0.0, 1.0, 4.0]),
    )
    npt.assert_almost_equal(actual, expected)


def test_create_xyz_points():
    actual = vk.create_xyz_points(ofk.sphere, [-1, 0, 1])
    expected = vk.XyzPoints(
        x=np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
        y=np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
        z=np.array([[2.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 2.0]]),
    )
    npt.assert_almost_equal(actual, expected)

    actual = vk.create_xyz_points(ofk.sphere, [-1, 0, 1], [2, 3, 4])
    expected = vk.XyzPoints(
        x=np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
        y=np.array([[2, 2, 2], [3, 3, 3], [4, 4, 4]]),
        z=np.array([[5.0, 4.0, 5.0], [10.0, 9.0, 10.0], [17.0, 16.0, 17.0]]),
    )
    npt.assert_almost_equal(actual, expected)


def test_discrete_cmap():
    n = 5
    rgba_list = vk.discrete_cmap(n)
    assert len(rgba_list) == n
    assert all(len(rgba) == 4 for rgba in rgba_list)
    assert all(isinstance(value, float) for rgba in rgba_list for value in rgba)


def test_plot_contour():
    actual = toolz.pipe(
        vk.create_xyz_points(ofk.sphere, [-1, 0, 1]),
        vk.plot_contour,
    )
    plt.close()
    assert isinstance(actual, matplotlib.axes.Axes)


def test_plot_line():
    actual = toolz.pipe(
        vk.create_xy_points(ofk.sphere, [-1, 0, 1]),
        vk.plot_line,
    )
    plt.close()
    assert isinstance(actual, matplotlib.axes.Axes)


def test_plot_surface():
    actual = toolz.pipe(
        vk.create_xyz_points(ofk.sphere, [-1, 0, 1]),
        vk.plot_surface,
    )
    plt.close()
    assert isinstance(actual, mpl_toolkits.mplot3d.axes3d.Axes3D)


def test_plot_xy_points():
    actual = toolz.pipe(
        vk.create_xy_points(ofk.sphere, [-1, 0, 1]),
        vk.plot_xy_points,
    )
    plt.close()
    assert isinstance(actual, matplotlib.axes.Axes)


def test_plot_xyz_points():
    actual = toolz.pipe(
        vk.create_xyz_points(ofk.sphere, [-1, 0, 1]),
        vk.plot_xyz_points,
    )
    plt.close()
    assert isinstance(actual, mpl_toolkits.mplot3d.axes3d.Axes3D)
