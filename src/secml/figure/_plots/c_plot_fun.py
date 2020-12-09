"""
.. module:: CPlotFunction
   :synopsis: Function plots.

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from secml.figure._plots import CPlot
from secml.figure._plots.plot_utils import create_points_grid
from secml.array import CArray
from secml.core.constants import inf
from secml.core.type_utils import is_list


class CPlotFunction(CPlot):
    """Plots a Function.

    Custom plotting parameters can be specified.

    Currently parameters default:
     - show_legend: True
     - grid: True

    See Also
    --------
    .CPlot : basic subplot functions.
    .CFigure : creates and handle figures.

    """

    def apply_params_fun(self):
        """Apply defined parameters to active subplot."""
        fig_legend = self.get_legend()
        if self.show_legend is not False and fig_legend is not None:
            fig_legend.set_visible(True)
        self.grid(grid_on=True)

    def plot_fun(self, func, multipoint=False,
                 plot_background=True, plot_levels=True,
                 levels=None, levels_color='k', levels_style=None,
                 levels_linewidth=1.0, n_colors=50, cmap='jet',
                 alpha=1.0, alpha_levels=1.0, vmin=None, vmax=None,
                 colorbar=True, n_grid_points=30,
                 grid_limits=None, func_args=(), **func_kwargs):
        """Plot a function (used for decision functions or boundaries).

        Parameters
        ----------
        func : unbound function
            Function to be plotted.
        multipoint : bool, optional
            If True, all grid points will be passed to the function.
            If False (default), function is iterated over each
            point of the grid.
        plot_background : bool, optional
            Specifies whether to plot the value of func at each point
            in the background using a colorbar.
        plot_levels : bool, optional
            Specify if function levels should be plotted (default True).
        levels : list or None, optional
            List of levels to be plotted.
            If None, 0 (zero) level will be plotted.
        levels_color : str or tuple or None, optional
            If None, the colormap specified by cmap will be used.
            If a string, like 'k', all levels will be plotted in this color.
            If a tuple of colors (string, float, rgb, etc),
            different levels will be plotted in different colors
            in the order specified. Default 'k'.
        levels_style : [ None | 'solid' | 'dashed' | 'dashdot' | 'dotted' ]
            If levels_style is None, the default is 'solid'.
            levels_style can also be an iterable of the above strings
            specifying a set of levels_style to be used. If this iterable
            is shorter than the number of contour levels it will be
            repeated as necessary.
        levels_linewidth : float or list of floats, optional
            The line width of the contour lines. Default 1.0.
        n_colors : int, optional
            Number of color levels of background plot. Default 50.
        cmap : str or list or `matplotlib.pyplot.cm`, optional
            Colormap to use (default 'jet'). Could be a list of colors.
        alpha : float, optional
            The alpha blending value of the background. Default 1.0.
        alpha_levels : float, optional
            The alpha blending value of the levels. Default 1.0.
        vmin, vmax : float or None, optional
            Limits of the colors used for function plotting.
            If None, colors are determined by the colormap.
        colorbar : bool, optional
            True if colorbar should be displayed.
        n_grid_points : int, optional
            Number of grid points.
        grid_limits : list of tuple, optional
            List with a tuple of min/max limits for each axis.
            If None, [(0, 1), (0, 1)] limits will be used.
        func_args, func_kwargs
            Other arguments or keyword arguments to pass to `func`.

        Examples
        --------
        .. plot:: pyplots/plot_fun.py
            :include-source:

        """
        levels = [0] if levels is None else levels

        # create the grid of the point where the function will be evaluated
        pad_grid_point_features, pad_xgrid, pad_ygrid = \
            create_points_grid(grid_limits, n_grid_points)

        # Evaluate function on each grid point
        if multipoint is True:
            grid_points_value = func(
                pad_grid_point_features, *func_args, **func_kwargs)
        else:
            grid_points_value = pad_grid_point_features.apply_along_axis(
                func, 1, *func_args, **func_kwargs)

        grid_points_val_reshaped = grid_points_value.reshape(
            (pad_xgrid.shape[0], pad_xgrid.shape[1]))

        # Clipping values to show a correct color plot
        clip_min = -inf if vmin is None else vmin
        clip_max = inf if vmax is None else vmax
        grid_points_val_reshaped = grid_points_val_reshaped.clip(
            clip_min, clip_max)

        if is_list(cmap):  # Convert list of colors to colormap
            from matplotlib.colors import ListedColormap
            cmap = ListedColormap(cmap)

        ch = None
        if plot_background is True:
            # Draw a fully colored plot using 50 levels
            ch = self.contourf(pad_xgrid, pad_ygrid,
                               grid_points_val_reshaped,
                               n_colors, cmap=cmap, alpha=alpha,
                               vmin=vmin, vmax=vmax, zorder=0)

            # Displaying 20 ticks on the colorbar
            if colorbar is True:
                some_y = CArray.linspace(
                    grid_points_val_reshaped.min(),
                    grid_points_val_reshaped.max(), 20)
                self.colorbar(ch, ticks=some_y)

        if plot_levels is True:
            self.contour(
                pad_xgrid, pad_ygrid, grid_points_val_reshaped,
                levels=levels, colors=levels_color, linestyles=levels_style,
                linewidths=levels_linewidth, alpha=alpha_levels)

        # Customizing figure
        self.apply_params_fun()

        return ch

    def plot_fgrads(self, gradf, n_grid_points=30, grid_limits=None,
                    color='k', linestyle='-', linewidth=1.0, alpha=1.0,
                    func_args=(), **func_kwargs):
        """Plot function gradient directions.

        Parameters
        ----------
        gradf : function
            Function that computes gradient directions.
        n_grid_points : int
            Number of grid points.
        grid_limits : list of tuple
            List with a tuple of min/max limits for each axis.
            If None, [(0, 1), (0, 1)] limits will be used.
        color :
            Color of the gradient directions.
        linestyle : str
            ['solid' | 'dashed', 'dashdot', 'dotted' |
            (offset, on-off-dash-seq) | '-' | '--' | '-.' | ':' |
            'None' | ' ' | '']
        linewidth : float
            Width of the line.
        alpha : float
            Transparency factor of the directions.
        func_args, func_kwargs : any
            Other arguments or keyword arguments to pass to `gradf`.

        """
        # create the grid of the point where the function will be evaluated
        pad_grid_point_features, pad_xgrid, pad_ygrid = \
            create_points_grid(grid_limits, n_grid_points)

        n_vals = pad_grid_point_features.shape[0]
        grad_point_values = CArray.zeros((n_vals, 2))
        # compute gradient on each grid point
        for p_idx in range(n_vals):
            grad_point_values[p_idx, :] = gradf(
                pad_grid_point_features[p_idx, :].ravel(),
                *func_args, **func_kwargs)

        U = grad_point_values[:, 0].reshape(
            (pad_xgrid.shape[0], pad_xgrid.shape[1]))
        V = grad_point_values[:, 1].reshape(
            (pad_xgrid.shape[0], pad_xgrid.shape[1]))

        self.quiver(U, V, pad_xgrid, pad_ygrid,
                    color=color, linestyle=linestyle,
                    linewidth=linewidth, alpha=alpha)

        # Customizing figure
        self.apply_params_fun()
