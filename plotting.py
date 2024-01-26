from typing import Optional, Sequence
from collections.abc import Iterable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from more_itertools import zip_broadcast

from node_homotopy.utils import cast_to_nparray


class _TrajectoryPlotter:
    def __init__(
        self,
        n_plots: int,
        single_axis: bool = False,
        labels: Optional[Iterable[str]] = None,
        xlabels: Optional[Iterable[str]] = None,
        ylabels: Optional[Iterable[str]] = None,
        fig: Optional[Figure] = None,
        figsize: tuple[float, float] = (10, 5),
    ):
        self.n_plots = int(n_plots)
        self.single_axis = single_axis

        # Might want to wrap these in another function or use inheritance or rewrite as pure functions imported from a different script
        self.labels = self._check_labels(labels)
        self.xlabels = self._check_axis_labels(xlabels)
        self.ylabels = self._check_axis_labels(ylabels)
        self.figure = self._check_figure(fig, figsize)

    def _check_labels(self, labels):
        if labels is None:
            labels = [None] * self.n_plots
        else:
            assert (
                len(labels) == self.n_plots
            ), "Number of labels must match the number of plots"
        return labels

    def _check_axis_labels(self, axis_labels):
        if axis_labels is None:
            axis_labels = [None] * self.n_axes
        else:
            assert (
                len(axis_labels) == self.n_axes
            ), "Number of x, y labels must match the number of axes"
        return axis_labels

    def _check_figure(self, fig, figsize):
        if fig is None:
            fig, _ = plt.subplots(1, self.n_axes, figsize=figsize)
        return fig

    @property
    def n_axes(self):
        return 1 if self.single_axis else self.n_plots

    @property
    def axes(self):
        axes_list = self.figure.axes
        return axes_list[0] if self.single_axis else axes_list

    def plot(self, t: np.ndarray, u: np.ndarray, **plot_kwargs):
        assert (
            u.shape[0] == self.n_plots
        ), "Degree of freedom of state vector u must match the number of plots"

        for i, (ax, label) in enumerate(zip_broadcast(self.axes, self.labels)):
            ax.plot(t, u[i], label=label, marker=".", markersize=3, **plot_kwargs)

        for ax, xlabel, ylabel in zip_broadcast(self.axes, self.xlabels, self.ylabels):
            ax.grid(ls="--", color="lightgray")
            ax.legend()
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)


def plot_trajectory(
    t,
    u,
    single_axis: bool = False,
    labels: Optional[Iterable[str]] = None,
    xlabels: Optional[Iterable[str]] = None,
    ylabels: Optional[Iterable[str]] = None,
    fig: Optional[Figure] = None,
    figsize: tuple[float, float] = (10, 5),
    **plot_kwargs
):
    t, u = cast_to_nparray(t), cast_to_nparray(u)
    n_plots = u.shape[0]

    plotter = _TrajectoryPlotter(
        n_plots, single_axis, labels, xlabels, ylabels, fig, figsize
    )
    plotter.plot(t, u, **plot_kwargs)
    return plotter.figure


def plot_line_and_band(
    ax,
    x,
    y_line,
    y_halfwidth,
    label: str | None = None,
    color: str | None = None,
    band_alpha: float = 0.2,
    line_alpha: float = 0.8,
    **line_kwargs
):
    ax.plot(x, y_line, label=label, color=color, alpha=line_alpha, **line_kwargs)
    ax.fill_between(
        x,
        y_line - y_halfwidth,
        y_line + y_halfwidth,
        color=color,
        alpha=band_alpha,
    )
    return ax


def waterfall_plot(
    ax,
    ypos: Sequence[float],
    xs: Sequence[np.ndarray],
    zs: Sequence[np.ndarray],
    inv_offset: float = 2.0,
    cmap: str = "plasma",
    alpha: float = 0.5,
    linewidth: float = 1.5,
    clabel: str | None = None,
):
    """Adapted from: https://stackoverflow.com/questions/55781132/imitating-the-waterfall-plots-in-origin-with-matplotlib"""
    assert (
        len(ypos) == len(xs) == len(zs)
    ), "The number of (x, z) pairs must be the same as the number of y positions!"

    for side in ["right", "top", "left"]:
        ax.spines[side].set_visible(False)

    z_max = max(np.amax(z) for z in zs)
    z_min = min(np.amin(z) for z in zs)
    z_range = z_max - z_min
    dy = (
        np.sqrt(abs(z_range)) / inv_offset
    )  # a tuning parameter for the offset of each dataset

    N_pos = len(ypos)
    for i, (x, z) in enumerate(zip(xs, zs)):
        y = z + i * dy  # the shifted z data used for plotting
        y_ind = N_pos - i  # used to set plot order

        # fill with white from the (shifted) y data down to the lowest value
        # for good results, don't make the alpha too low, otherwise you'll get confusing blending of lines
        ax.fill_between(x, z_min, y, facecolor="white", alpha=alpha, zorder=y_ind)

        # cut the data into segments that can be colored individually
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a continuous norm to map from data points to colors
        norm = plt.Normalize(z_min, z_max)
        lc = LineCollection(segments, cmap=cmap, norm=norm)

        # Set the values used for colormapping
        lc.set_array(z)
        lc.set_zorder(y_ind)
        lc.set_linewidth(linewidth)
        line = ax.add_collection(lc)

    # set limits, as using LineCollection does not automatically set these
    ax.set_ylim(z_min, z_max + N_pos * dy)
    # ax.set_xlim(-10, 10)
    plt.yticks([])
    ax.yaxis.set_ticks_position("none")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    ax.get_figure().colorbar(line, ax=ax, cax=cax, label=clabel)

    return ax


def plot_surface(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    z_clip_lim: tuple[float, float] | None = None,
    cmap: str = "plasma",
):
    z_ = z if z_clip_lim is None else np.clip(z, *z_clip_lim)
    ax.plot_surface(*np.meshgrid(x, y), z_, cmap=cmap)

    # Remove the background from the xy, yz, zx planes in the background
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Add silver dashed grid lines
    ax.xaxis._axinfo["grid"].update({"linestyle": "--", "color": "silver"})
    ax.yaxis._axinfo["grid"].update({"linestyle": "--", "color": "silver"})
    ax.zaxis._axinfo["grid"].update({"linestyle": "--", "color": "silver"})

    return ax
