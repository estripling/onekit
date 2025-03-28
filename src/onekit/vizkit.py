from enum import Enum
from typing import (
    Callable,
    NamedTuple,
)

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d.axes3d import Axes3D

from onekit import numpykit as npk
from onekit import pythonkit as pk

__all__ = (
    "Config",
    "FunctionPlotter",
    "create_xy_points",
    "create_xyz_points",
    "discrete_cmap",
    "plot_contour",
    "plot_digitscale",
    "plot_line",
    "plot_surface",
    "plot_xy_points",
    "plot_xyz_points",
)

ArrayLike = npt.ArrayLike
Matrix = npt.NDArray[npt.NDArray[np.float64]]
Pair = tuple[float, float]
RGBA = tuple[float, float, float, float]
Vector = npt.NDArray[np.float64]

Func1n = Callable[[float], float]
Func2n = Callable[[Vector], float]


class Config(Enum):
    """Configurations for visualization."""

    @classmethod
    def get_kws_contour__base(cls) -> dict:
        """Returns kwargs for ``.contour()``: base configuration."""
        return dict(
            levels=12,
            colors="dimgray",
            antialiased=True,
            linewidths=0.25,
            alpha=1.0,
            zorder=1,
        )

    @classmethod
    def get_kws_contourf__base(cls) -> dict:
        """Returns kwargs for ``.contourf()``: base configuration."""
        return dict(
            levels=100,
            antialiased=True,
            alpha=0.61803,
            zorder=0,
        )

    @classmethod
    def get_kws_contourf__YlOrBr(cls) -> dict:
        """Returns kwargs for ``.contourf()``:
        ``YlOrBr`` configuration for dark max.
        """
        kws = cls.get_kws_contourf__base()
        additional_kws = dict(
            cmap=plt.get_cmap("YlOrBr"),
        )
        kws.update(additional_kws)
        return kws

    @classmethod
    def get_kws_contourf__YlOrBr_r(cls) -> dict:
        """Returns kwargs for ``.contourf()``:
        ``YlOrBr_r`` configuration for dark min.
        """
        kws = cls.get_kws_contourf__base()
        additional_kws = dict(
            cmap=plt.get_cmap("YlOrBr_r"),
        )
        kws.update(additional_kws)
        return kws

    @classmethod
    def get_kws_plot__base(cls) -> dict:
        """Returns kwargs for ``.plot()``: base configuration."""
        return dict(
            linewidth=2,
            zorder=0,
        )

    @classmethod
    def get_kws_scatter__base(cls) -> dict:
        """Returns kwargs for ``.scatter()``: base configuration."""
        return dict(
            s=60,
            c="black",
            zorder=2,
        )

    @classmethod
    def get_kws_surface__base(cls) -> dict:
        """Returns kwargs for ``.plot_surface()``: base configuration."""
        return dict(
            rstride=2,
            cstride=2,
            edgecolors="dimgray",
            antialiased=True,
            linewidth=0.1,
            alpha=0.61803,
            zorder=0,
        )

    @classmethod
    def get_kws_surface__YlOrBr(cls) -> dict:
        """Returns kwargs for ``.plot_surface()``:
        ``YlOrBr`` configuration for dark max.
        """
        kws = cls.get_kws_surface__base()
        additional_kws = dict(
            cmap=plt.get_cmap("YlOrBr"),
        )
        kws.update(additional_kws)
        return kws

    @classmethod
    def get_kws_surface__YlOrBr_r(cls) -> dict:
        """Returns kwargs for ``.plot_surface()``:
        ``YlOrBr_r`` configuration for dark min.
        """
        kws = cls.get_kws_surface__base()
        additional_kws = dict(
            cmap=plt.get_cmap("YlOrBr_r"),
        )
        kws.update(additional_kws)
        return kws


class Point(NamedTuple):
    """Two- or three-dimensional coordinate point."""

    x: float
    y: float
    z: float | None = None


class XyPoints(NamedTuple):
    """(x, y) coordinate points."""

    x: Vector
    y: Vector


class XyzPoints(NamedTuple):
    """(x, y, z) coordinate points."""

    x: Vector | Matrix
    y: Vector | Matrix
    z: Vector | Matrix


class FunctionPlotter:
    """Plot :math:`f(x)` versus coordinate vector :math:`x`.

    Examples
    --------
    >>> from onekit import optfunckit as ofk
    >>> from onekit import vizkit as vk
    >>> plotter = vk.FunctionPlotter(ofk.sphere, bounds=[(-5, 5)])
    >>> plotter
    FunctionPlotter(func=sphere, bounds=[(-5, 5)])
    >>> plotter.plot()  # doctest: +SKIP
    """

    def __init__(
        self,
        func: Func1n | Func2n,
        bounds: list[Pair],
        *,
        with_contour: bool = True,
        with_surface: bool = True,
        n_xvalues: int = 101,
        x1_values: ArrayLike | None = None,
        x2_values: ArrayLike | None = None,
        points: list[Point] | None = None,
        kws_contour=None,
        kws_contourf=None,
        kws_plot=None,
        kws_scatter=None,
        kws_surface=None,
    ):
        self._func = func
        self._bounds = bounds
        self._with_contour = with_contour
        self._with_surface = with_surface
        self._n_xvalues = n_xvalues
        self._x1_values = x1_values
        self._x2_values = x2_values
        self._points = points
        self._kws_contour = kws_contour
        self._kws_contourf = kws_contourf
        self._kws_plot = kws_plot
        self._kws_scatter = kws_scatter
        self._kws_surface = kws_surface

        self._n = len(self.bounds)
        self._xyz_pts = None

        if self._n not in (1, 2):
            raise TypeError("the total number of bounds must be either 1 or 2")

        if self._n == 2 and with_surface is False and with_contour is False:
            raise ValueError("`with_surface` and `with_contour` cannot both be False")

    def __repr__(self):
        return f"{type(self).__name__}(func={self.func.__name__}, bounds={self.bounds})"

    @property
    def func(self) -> Func1n | Func2n:
        """Function to plot."""
        return self._func

    @property
    def bounds(self) -> list[Pair]:
        """Bounds to use in the plot."""
        return self._bounds

    def plot(self, fig=None, ax=None, ax3d=None) -> tuple[Figure, Axes, Axes3D]:
        """Make plot.

        Notes
        -----
        When creating both a surface and contour plot and either
        ``ax`` or ``ax3d`` is specified, it is best to also specify ``fig``.
        To this end, it might be easier to only specify a ``fig`` object.
        """
        self._set_xyz_pts()

        if self._n == 1:
            fig, ax, ax3d = self._plot_line(fig, ax, ax3d)

        else:
            if self._with_surface and self._with_contour:
                fig, ax, ax3d = self._plot_surface_and_contour(fig, ax, ax3d)

            if self._with_surface and not self._with_contour:
                fig, ax, ax3d = self._plot_surface(fig, ax, ax3d)

            if not self._with_surface and self._with_contour:
                fig, ax, ax3d = self._plot_contour(fig, ax, ax3d)

        if self._points is not None:
            ax, ax3d = self._add_points(ax, ax3d)

        return fig, ax, ax3d

    def _plot_surface_and_contour(self, fig, ax, ax3d):
        fig = fig or plt.figure(figsize=plt.figaspect(0.4))

        ax3d = ax3d or fig.add_subplot(1, 2, 1, projection="3d")
        ax3d = plot_surface(
            self._xyz_pts,
            kws_surface=self._kws_surface,
            kws_contourf=self._kws_contourf,
            ax=ax3d,
        )

        ax = ax or fig.add_subplot(1, 2, 2)
        ax = plot_contour(
            self._xyz_pts,
            kws_contourf=self._kws_contourf,
            kws_contour=self._kws_contour,
            ax=ax,
        )
        ax.axis("scaled")

        return fig, ax, ax3d

    def _plot_surface(self, fig, ax, ax3d):
        fig = fig or plt.gcf()

        ax3d = ax3d or fig.add_subplot(1, 1, 1, projection="3d")
        ax3d = plot_surface(
            self._xyz_pts,
            kws_surface=self._kws_surface,
            kws_contourf=self._kws_contourf,
            ax=ax3d,
        )

        ax = None

        return fig, ax, ax3d

    def _plot_contour(self, fig, ax, ax3d):
        fig = fig or plt.gcf()

        ax3d = None

        ax = ax or fig.add_subplot(1, 1, 1)
        ax = plot_contour(
            self._xyz_pts,
            kws_contourf=self._kws_contourf,
            kws_contour=self._kws_contour,
            ax=ax,
        )
        ax.axis("scaled")

        return fig, ax, ax3d

    def _plot_line(self, fig, ax, ax3d):
        fig = fig or plt.gcf()

        ax3d = None

        ax = ax or fig.add_subplot(1, 1, 1)
        ax = plot_line(
            self._xyz_pts,
            kws_plot=self._kws_plot,
            ax=ax,
        )

        return fig, ax, ax3d

    def _add_points(self, ax, ax3d):
        x = npk.check_vector([p.x for p in self._points])
        y = npk.check_vector([p.y for p in self._points])

        if ax is not None:
            xy_pts = XyPoints(x, y)
            ax = plot_xy_points(xy_pts, kws_scatter=self._kws_scatter, ax=ax)

        if ax3d is not None:
            z = npk.check_vector([p.z for p in self._points])
            xyz_pts = XyzPoints(x, y, z)
            ax3d = plot_xyz_points(xyz_pts, kws_scatter=self._kws_scatter, ax=ax3d)

        return ax, ax3d

    def _set_xyz_pts(self):
        """Private setter for (x, y, z) coordinate points."""
        if self._xyz_pts is None:
            if self._n == 1:
                (x_bounds,) = self.bounds
                xmin, xmax = min(x_bounds), max(x_bounds)
                self._xyz_pts = create_xy_points(
                    self._func,
                    self._x1_values or np.linspace(xmin, xmax, self._n_xvalues),
                )

            else:
                x_bounds, y_bounds = self.bounds
                xmin, xmax = min(x_bounds), max(x_bounds)
                ymin, ymax = min(y_bounds), max(y_bounds)
                self._xyz_pts = create_xyz_points(
                    self._func,
                    self._x1_values or np.linspace(xmin, xmax, self._n_xvalues),
                    self._x2_values or np.linspace(ymin, ymax, self._n_xvalues),
                )


def create_xy_points(func: Func1n, x_values: ArrayLike, /) -> XyPoints:
    """Compute :math:`(x, y)` coordinate points.

    Examples
    --------
    >>> from onekit import optfunckit as ofk
    >>> from onekit import vizkit as vk
    >>> vk.create_xy_points(ofk.sphere, [-2, -1, 0, 1, 2])
    XyPoints(x=array([-2, -1,  0,  1,  2]), y=array([4., 1., 0., 1., 4.]))
    """
    x = npk.check_vector(x_values, n_min=2)
    y = np.apply_along_axis(func1d=func, axis=1, arr=np.c_[x.ravel()])
    return XyPoints(x, y)


def create_xyz_points(
    func: Func2n,
    x_values: ArrayLike,
    y_values: ArrayLike | None = None,
    /,
) -> XyzPoints:
    """Compute :math:`(x, y, z)` coordinate points.

    Examples
    --------
    >>> from onekit import optfunckit as ofk
    >>> from onekit import vizkit as vk
    >>> vk.create_xyz_points(ofk.sphere, [-1, 0, 1])
    XyzPoints(x=array([[-1,  0,  1],
           [-1,  0,  1],
           [-1,  0,  1]]), y=array([[-1, -1, -1],
           [ 0,  0,  0],
           [ 1,  1,  1]]), z=array([[2., 1., 2.],
           [1., 0., 1.],
           [2., 1., 2.]]))
    """
    x_values = npk.check_vector(x_values, n_min=2)
    y_values = x_values if y_values is None else npk.check_vector(y_values, n_min=2)
    x, y = np.meshgrid(x_values, y_values)
    z = np.apply_along_axis(func1d=func, axis=1, arr=np.c_[x.ravel(), y.ravel()])
    return XyzPoints(x, y, z.reshape(x.shape))


def discrete_cmap(
    n: int,
    /,
    *,
    name: str = "viridis_r",
    lower_bound: float = 0.05,
    upper_bound: float = 0.9,
) -> list[RGBA]:
    """Create discrete colormap values.

    Examples
    --------
    >>> import numpy as np
    >>> from onekit import vizkit as vk
    >>> np.set_printoptions(legacy="1.21")
    >>> vk.discrete_cmap(2)
    [(0.876168, 0.891125, 0.09525, 1.0), (0.282623, 0.140926, 0.457517, 1.0)]
    """
    cmap = plt.get_cmap(name)
    return [cmap(i) for i in np.linspace(lower_bound, upper_bound, num=n)]


def plot_contour(
    xyz_pts: XyzPoints,
    /,
    *,
    kws_contour=None,
    kws_contourf=None,
    ax=None,
) -> Axes:
    """Plot :math:`z` versus :math:`(x, y)` as contour.

    Parameters
    ----------
    xyz_pts : XyzPoints
        :math:`(x, y, z)` coordinate points to plot.
    kws_contour : dict of keyword arguments, optional
        Keyword arguments to pass to ``matplotlib.axes.Axes.contour``.
        Default: ``VizConfig.get_kws_contour__base()``.
        Specify dict of keyword arguments to update configurations.
    kws_contourf : dict of keyword arguments, optional
        Keyword arguments to pass to ``matplotlib.axes.Axes.contourf``.
        Default: `VizConfig.get_kws_contourf__YlOrBr_r()``.
        Specify dict of keyword arguments to update configurations.
    ax : matplotlib.axes.Axes, optional
        Specify ``Axes`` object. Default: current ``Axes`` object.

    Examples
    --------
    >>> import toolz
    >>> from onekit import optfunckit as ofk
    >>> from onekit import vizkit as vk
    >>> ax = toolz.pipe(  # doctest: +SKIP
    ...     vk.create_xyz_points(ofk.sphere, [-1, 0, 1]),
    ...     vk.plot_contour,
    ... )
    """
    ax = ax or plt.gca()

    # plot filled contour first
    kwargs_contourf = Config.get_kws_contourf__YlOrBr_r()
    kwargs_contourf.update(kws_contourf or dict())
    contour_plot = ax.contourf(xyz_pts.x, xyz_pts.y, xyz_pts.z, **kwargs_contourf)

    # plot contour second
    kwargs_contour = Config.get_kws_contour__base()
    kwargs_contour.update(kws_contour or dict())
    ax.contour(xyz_pts.x, xyz_pts.y, xyz_pts.z, **kwargs_contour)

    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.15)
    plt.colorbar(contour_plot, cax=cax)

    return ax


def plot_digitscale(
    x,
    /,
    *,
    cmap_name: str = "YlOrBr_r",
    kws_plot: dict[str, str] | None = None,
    ax=None,
) -> Axes:
    """Plot a scaled version of :math:`x` using digitscale.

    See Also
    --------
    onekit.mathkit.digitscale : Python version

    Examples
    --------
    >>> from onekit import mathkit as mk
    >>> from onekit import vizkit as vk
    >>> vk.plot_digitscale(tuple(mk.collatz(27)))  # doctest: +SKIP
    """
    ax = ax or plt.gca()

    kwargs_plot = dict(marker=".", color="black")
    kwargs_plot.update(kws_plot or dict())

    x_values = np.asarray(x)
    x_scaled = npk.digitscale(x_values)
    ax.plot(x_scaled, **kwargs_plot)
    x_limits = ax.get_xlim()

    y_min = int(np.floor(x_scaled.min()))
    y_max = int(np.ceil(x_scaled.max()))
    y_range = tuple(range(y_min, y_max))
    colors = discrete_cmap(y_max, name=cmap_name, lower_bound=0.2, upper_bound=0.8)

    for i, y in enumerate(y_range):
        ax.fill_between(x_limits, y, y + 1, color=colors[i], alpha=0.3)

    ax.set_xlim(x_limits)
    ax.set_yticks(y_range)

    ax.set_xlabel("index")
    ax.set_ylabel("num_digits(orig_value)")

    ax.set_title(
        pk.concat_strings(
            ", ",
            f"mean={pk.num_to_str(x_values.mean())}",
            f"median={pk.num_to_str(float(np.median(x_values)))}",
            f"std={pk.num_to_str(x_values.std(ddof=1))}",
            f"min={pk.num_to_str(x_values.min())}",
            f"max={pk.num_to_str(x_values.max())}",
        ),
        size="medium",
        pad=12,
    )

    return ax


def plot_line(xy_pts: XyPoints, /, *, kws_plot=None, ax=None) -> Axes:
    """Plot :math:`y` versus :math:`x` as line.

    Parameters
    ----------
    xy_pts : XyPoints
        :math:`(x, y)` coordinate points to plot.
    kws_plot : dict of keyword arguments, optional
        Keyword arguments to pass to ``matplotlib.axes.Axes.plot``.
        Default: ``VizConfig.get_kws_plot__base()``.
        Specify dict of keyword arguments to update configurations.
    ax : matplotlib.axes.Axes, optional
        Specify ``Axes`` object. Default: current ``Axes`` object.

    Examples
    --------
    >>> import toolz
    >>> from onekit import optfunckit as ofk
    >>> from onekit import vizkit as vk
    >>> ax = toolz.pipe(  # doctest: +SKIP
    ...     vk.create_xy_points(ofk.sphere, [-2, -1, 0, 1, 2]),
    ...     vk.plot_line,
    ... )
    """
    ax = ax or plt.gca()

    kwargs_plot = Config.get_kws_plot__base()
    kwargs_plot.update(kws_plot or dict())
    ax.plot(xy_pts.x, xy_pts.y, **kwargs_plot)

    return ax


def plot_surface(
    xyz_pts: XyzPoints,
    /,
    *,
    kws_surface=None,
    kws_contourf=None,
    ax=None,
) -> Axes3D:
    """Plot :math:`z` versus :math:`(x, y)` as surface.

    Parameters
    ----------
    xyz_pts : XyzPoints
        :math:`(x, y, z)` coordinate points to plot.
    kws_surface : dict of keyword arguments, optional
        Keyword arguments to pass to
        ``mpl_toolkits.mplot3d.axes3d.Axes3D.plot_surface``.
        Default: ``VizConfig.get_kws_surface__YlOrBr_r()``.
        Specify dict of keyword arguments to update configurations.
    kws_contourf : dict of keyword arguments, optional
        Keyword arguments to pass to ``matplotlib.axes.Axes.contourf``.
        Default: ``VizConfig.get_kws_contourf__YlOrBr_r()``.
        Specify dict of keyword arguments to update configurations.
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D, optional
        Specify ``Axes3D`` object. Default: current ``Axes3D`` object.

    Examples
    --------
    >>> import toolz
    >>> from onekit import optfunckit as ofk
    >>> from onekit import vizkit as vk
    >>> ax = toolz.pipe(  # doctest: +SKIP
    ...     vk.create_xyz_points(ofk.sphere, [-1, 0, 1]),
    ...     vk.plot_surface,
    ... )
    """
    ax = ax or plt.gcf().add_subplot(projection="3d")

    # make background and axis panes transparent
    ax.patch.set_alpha(0.0)
    ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

    # plot surface first
    kwargs_surface = Config.get_kws_surface__YlOrBr_r()
    kwargs_surface.update(kws_surface or dict())
    ax.plot_surface(xyz_pts.x, xyz_pts.y, xyz_pts.z, **kwargs_surface)

    # plot filled contour second
    kwargs_contourf = Config.get_kws_contourf__YlOrBr_r()
    kwargs_contourf.update(kws_contourf or dict())
    kwargs_contourf["zdir"] = kwargs_contourf.get("zdir", "z")

    # plot contourf onto XY-plane.
    ax.set_zlim3d(xyz_pts.z.min(), xyz_pts.z.max())
    kwargs_contourf["offset"] = xyz_pts.z.min()

    ax.contourf(xyz_pts.x, xyz_pts.y, xyz_pts.z, **kwargs_contourf)

    return ax


def plot_xy_points(xy_pts: XyPoints, /, *, kws_scatter=None, ax=None) -> Axes:
    """Plot :math:`y` versus :math:`x` as scatter points.

    Parameters
    ----------
    xy_pts : XyPoints
        :math:`(x, y)` coordinate points to plot.
    kws_scatter : dict of keyword arguments, optional
        Keyword arguments to pass to ``matplotlib.axes.Axes.scatter``.
        Default: ``VizConfig.get_kws_scatter__base()``.
        Specify dict of keyword arguments to update configurations.
    ax : matplotlib.axes.Axes, optional
        Specify ``Axes`` object. Default: current ``Axes`` object.

    Examples
    --------
    >>> import toolz
    >>> from onekit import optfunckit as ofk
    >>> from onekit import vizkit as vk
    >>> ax = toolz.pipe(  # doctest: +SKIP
    ...     vk.create_xy_points(ofk.sphere, [-2, -1, 0, 1, 2]),
    ...     vk.plot_xy_points,
    ... )
    """
    ax = ax or plt.gca()

    kwargs_scatter = Config.get_kws_scatter__base()
    kwargs_scatter.update(kws_scatter or dict())
    ax.scatter(xy_pts.x, xy_pts.y, **kwargs_scatter)

    return ax


def plot_xyz_points(xyz_pts: XyzPoints, /, *, kws_scatter=None, ax=None) -> Axes3D:
    """Plot :math:`z` versus :math:`(x, y)` as scatter points.

    Parameters
    ----------
    xyz_pts : XyzPoints
        :math:`(x, y, z)` coordinate points to plot.
    kws_scatter : dict of keyword arguments, optional
        Keyword arguments to pass to
        ``mpl_toolkits.mplot3d.axes3d.Axes3D.scatter``.
        Default: ``VizConfig.get_kws_scatter__base()``.
        Specify dict of keyword arguments to update configurations.
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D, optional
        Specify ``Axes3D`` object. Default: current ``Axes3D`` object.

    Examples
    --------
    >>> import toolz
    >>> from onekit import optfunckit as ofk
    >>> from onekit import vizkit as vk
    >>> ax = toolz.pipe(  # doctest: +SKIP
    ...     vk.create_xyz_points(ofk.sphere, [-1, 0, 1]),
    ...     vk.plot_xyz_points,
    ... )
    """
    ax = ax or plt.gcf().add_subplot(projection="3d")

    # make background and axis panes transparent
    ax.patch.set_alpha(0.0)
    ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

    kwargs_scatter = Config.get_kws_scatter__base()
    kwargs_scatter.update(kws_scatter or dict())

    ax.scatter(xyz_pts.x, xyz_pts.y, ax.get_zlim()[0], **kwargs_scatter)
    ax.scatter(xyz_pts.x, xyz_pts.y, xyz_pts.z, **kwargs_scatter)

    return ax
