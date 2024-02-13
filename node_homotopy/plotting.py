# ruff: noqa: F722
from typing import Sequence

from jaxtyping import Float
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable

from node_homotopy.typealiases import ArrayOrTensor


def plot_line_and_band(
    ax: Axes,
    x: Float[ArrayOrTensor, " data"],
    y_line: Float[ArrayOrTensor, " data"],
    y_halfwidth: Float[ArrayOrTensor, " data"],
    label: str | None = None,
    color: str | None = None,
    line_alpha: float = 0.8,
    band_alpha: float = 0.2,
    **line_kwargs,
) -> Axes:
    """Plots y_line as a function of x, and shades the area between y_line-y_halfwidth and y_line+y_halfwidth.

    This is useful when trying to plot both the y as a function of x, and its margin of error.

    Args:
        ax: An matplotlib Axes to plot into.
        x: 1D numpy array or torch Tensor containing the x values.
            Has shape (number of data, ).
        y_line: 1D numpy array or torch Tensor containing the y values.
            Has shape (number of data, ).
        y_halfwidth: 1D numpy array or torch Tensor containing the half-width of the area to be shaded around y.
            Has shape (number of data, ).
        label: String denoting the label to be attached to the plot.
        color: String denoting the color for the plot.
        line_alpha: Float denoting the alpha value (transparency) of the line plot.
        band_alpha: Float denoting the alpha value (transparency) of the shaded area.
        **line_kwargs: Additional keyword arguments to be passed to ax.plot().

    Returns:
        ax: The matplotlib Axes mutated with the plot contents.
    """
    ax.plot(x, y_line, label=label, color=color, alpha=line_alpha, **line_kwargs)
    ax.fill_between(
        x,
        y_line - y_halfwidth,
        y_line + y_halfwidth,
        color=color,
        alpha=band_alpha,
    )
    return ax


def plot_waterfall(
    ax: Axes,
    ypos: Sequence[float],
    xs: Sequence[np.ndarray],
    zs: Sequence[np.ndarray],
    inv_offset: float = 2.0,
    cmap: str = "plasma",
    alpha: float = 0.5,
    linewidth: float = 1.5,
    clabel: str | None = None,
) -> Axes:
    """Plots a waterfall plot, such as the ones in Figure 2 of the paper.

    This is useful when trying to plot multiple plot of z versus x, that continuously change according to a third parameter y.
    Waterfall plots display multple z-x plots, ordered and stacked accordinging to each plot's y value.
    The code is adapted from: https://stackoverflow.com/questions/55781132/imitating-the-waterfall-plots-in-origin-with-matplotlib

    Args:
        ax: An matplotlib Axes to plot into.
        ypos: A sequence of y values to order the z-x plots with.
        xs: A sequence of numpy arrays of x, corresponding to the x values for each plot in the waterfall plot.
        zs: A sequence of numpy arrays of z, corresponding to the z values for each plot in the waterfall plot.
        inv_offset: Float denoting the the inverse of the offset between the stacked plots.
            This is a tuning parameter for the plot aesthetics. Larger values lead to more overlaps between the plots in the stack.
        cmap: String denoting the colormap to be used to denote the magnitude of z.
        alpha: Float denoting the alpha (transparency) value of the plots in the stack.
        linewidth: Float denoting the linewidth of each plot in the stack.
        clabel: String containing the label for the colorbar for the plot.

    Returns:
        ax: The matplotlib Axes mutated with the plot contents.
    """
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


Axes3D = (
    Axes  # Currently importing mpl_toolkits.mplot3d.axes3d.Axes3D causes an ImportError
)


def plot_surface(
    ax: Axes3D,
    x: Float[ArrayOrTensor, " grid_x"],
    y: Float[ArrayOrTensor, " grid_y"],
    z: Float[ArrayOrTensor, "grid_x grid_y"],
    z_clip_lim: tuple[float, float] | None = None,
    cmap: str = "plasma",
) -> Axes3D:
    """Plots a 3D surface plot of z(x, y); that is z as a function of x and y.

    Args:
        ax: The mpl_toolkit 3D axes object to plot into.
        x: 1D numpy array or torch Tensor containing the x grid values.
            Has shape (number of x grid, ).
        y: 1D numpy array or torch Tensor containing the y grid values.
            Has shape (number of y grid, ).
        z: 2D numpy array or torch Tensor containing the surface height values.
            Has shape (number of x grid, number of y_grid).
        z_clip_lim: An optional tuple of floats containing the upper and lower bounds to clip the surface with.
            This is useful in cases where the surface has very high/low regions and plotting the entire surface
            without clipping results in a very distorted plot.
        cmap: String denoting the colormap to use for the plot.

    Returns:
        ax: The mpl_toolkit 3D axes object mutated with the plot contents.
    """
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
