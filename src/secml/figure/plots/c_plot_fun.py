from secml.figure.plots import CPlot
from secml.array import CArray
from secml.core.constants import inf


class CPlotFunction(CPlot):
    """Plots a Function.

    Custom plotting parameters can be specified.
    Currently parameters default:
     - show_legend: True. Set False to hide legend on next plot.
     - grid: True.

    Parameters
    ----------
    sp : Axes
        Subplot to use for plotting. Instance of `matplotlib.axes.Axes`.
    default_params : dict
        Dictionary with default parameters.

    See Also
    --------
    .CPlot : basic subplot functions.
    .CFigure : creates and handle figures.

    """
    class_type = 'function'

    def __init__(self, sp, default_params=None):

        # Calling CPlot constructor
        super(CPlotFunction, self).__init__(
            sp=sp, default_params=default_params)

        # Specific plot parameters (use `set_params` to alter)
        self.show_legend = True
        self.grid(grid_on=True)

    def _apply_params(self):
        """Apply defined parameters to active subplot."""
        fig_legend = self.get_legend()
        if fig_legend is not None:
            fig_legend.set_visible(self.show_legend)

    @staticmethod
    def _create_points_grid(grid_limits, n_grid_points):
        """Creates a grid of points.

        Parameters
        ----------
        grid_limits : list of tuple
            List with a tuple of min/max limits for each axis.
            If None, [(0, 1), (0, 1)] limits will be used.
        n_grid_points : int
            Number of grid points.

        """
        grid_bounds = [(0, 1), (0, 1)] if grid_limits is None else grid_limits
        x_min, x_max = (grid_bounds[0][0], grid_bounds[0][1])
        y_min, y_max = (grid_bounds[1][0], grid_bounds[1][1])

        # Padding x and y grid points
        padding_x, padding_y = (0.05 * (x_max - x_min), 0.05 * (y_max - y_min))
        # Create the equi-spaced indices for each axis
        x_grid_points = CArray.linspace(
            x_min - padding_x, x_max + padding_x, num=n_grid_points)
        y_grid_points = CArray.linspace(
            y_min - padding_y, y_max + padding_y, num=n_grid_points)
        # Create the grid
        pad_xgrid, pad_ygrid = CArray.meshgrid((x_grid_points, y_grid_points))
        pad_grid_point_features = CArray.concatenate(
            pad_xgrid.reshape((pad_xgrid.size, 1)),
            pad_ygrid.reshape((pad_ygrid.size, 1)), axis=1)

        return pad_grid_point_features, pad_xgrid, pad_ygrid

    def plot_fobj(self, func, multipoint=False,
                  plot_background=True, plot_levels=True,
                  levels=None, levels_color='k', levels_style=None,
                  n_colors=50, cmap='jet', alpha=1.0, alpha_levels=1.0,
                  vmin=None, vmax=None, colorbar=True, n_grid_points=30,
                  grid_limits=None,  func_args=(), **func_kwargs):
        """Plot a function (used for discriminant functions or boundaries).

        Parameters
        ----------
        func : unbound function
            Function to be plotted.
        multipoint : bool
            If True, all grid points will be passed to the function.
            If False (default), function is iterated over each
            point of the grid.
        plot_background : bool
            Specifies whether to plot the value of func at each point
            in the background using a colorbar.
        plot_levels : bool
            Specify if function levels should be plotted (default True).
        levels : list
            List of levels to be plotted.
            If None, 0 (zero) level will be plotted.
        levels_color : str or tuple or None (default 'k')
            If None, the colormap specified by cmap will be used.
            If a string, like 'k', all levels will be plotted in this color.
            If a tuple of colors (string, float, rgb, etc),
            different levels will be plotted in different colors
            in the order specified.
        levels_style: [ None | 'solid' | 'dashed' | 'dashdot' | 'dotted' ]
            If levels_style is None, the default is 'solid'.
            levels_style can also be an iterable of the above strings
            specifying a set of levels_style to be used. If this iterable
            is shorter than the number of contour levels it will be
            repeated as necessary.
        n_colors : int
            Number of color levels of background plot. Default 50.
        cmap : str
            Colormap to use (default 'jet').
        alpha : float
            The alpha blending value of the background. Default 1.0.
        alpha_levels : float
            The alpha blending value of the levels. Default 1.0.
        vmin, vmax : float or None
            Limits of the colors used for function plotting.
            If None, colors are determined by the colormap.
        colorbar : bool
            True if colorbar should be displayed.
        n_grid_points : int
            Number of grid points.
        grid_limits : list of tuple
            List with a tuple of min/max limits for each axis.
            If None, [(0, 1), (0, 1)] limits will be used.
        func_args, func_kwargs : any
            Other arguments or keyword arguments to pass to `func`.

        Examples
        --------
        .. plot:: pyplots/plot_fobj.py
            :include-source:

        """
        levels = [0] if levels is None else levels

        # create the grid of the point where the function will be evaluated
        pad_grid_point_features, pad_xgrid, pad_ygrid = \
            self._create_points_grid(grid_limits, n_grid_points)

        # Evaluate function on each grid point
        if multipoint is True:
            grid_points_value = func(
                pad_grid_point_features, *func_args, **func_kwargs)
        else:
            grid_points_value = pad_grid_point_features.apply_fun_torow(
                func, *func_args, **func_kwargs)

        grid_points_val_reshaped = grid_points_value.reshape(
            (pad_xgrid.shape[0], pad_xgrid.shape[1]))

        # Clipping values to show a correct color plot
        clip_min = -inf if vmin is None else vmin
        clip_max = inf if vmax is None else vmax
        grid_points_val_reshaped = grid_points_val_reshaped.clip(
            clip_min, clip_max)

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
                self.colorbar(ch, cmap=cmap, ticks=some_y)

        if plot_levels is True:
            self.contour(
                pad_xgrid, pad_ygrid, grid_points_val_reshaped,
                levels=levels, colors=levels_color, linestyles=levels_style,
                alpha=alpha_levels)

        # Customizing figure
        self._apply_params()

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
            self._create_points_grid(grid_limits, n_grid_points)

        n_vals = pad_grid_point_features.shape[0]
        grad_point_values = CArray.zeros((n_vals, 2))
        # compute gradient on each grid point
        for p_idx in xrange(n_vals):
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
        self._apply_params()
