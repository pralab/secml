"""
.. module:: CPlot
   :synopsis: A standard plot.

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
import inspect
import sys

from matplotlib.axes import Axes

from secml.core import CCreator
from secml.array import CArray
from secml.array.array_utils import tuple_sequence_tondarray


class CPlot(CCreator):
    """Interface for standard plots.

    This class provides an interface and few other methods useful
    for standard plot creation.

    To be never explicitly instanced. Will be created by `CFigure`.

    Parameters
    ----------
    sp : Axes
        Subplot to use for plotting. Instance of `matplotlib.axes.Axes`.
    default_params : dict
        Dictionary with default parameters.

    See Also
    --------
    .CFigure : creates and handle figures.

    """

    def __init__(self, sp, default_params):

        if not isinstance(sp, Axes):
            raise TypeError("`matplotlib.axes.Axes` instance is requested.")

        # Target subplot reference
        self._sp = sp
        # Store default parameters
        self._params = default_params
        # Collect methods from subclasses
        self._collect_spmethods()

        # Callback parameter for showing the legend after
        # applying custom plot parameters
        self.show_legend = None

        # Placeholders for plot parameters
        self._ylabel = None
        self._xlabel = None
        self._yticks = None
        self._yticklabels = None
        self._xticks = None
        self._xticklabels = None
        self._xlim = None
        self._ylim = None

    def _collect_spmethods(self):
        """Collects methods from CPlot subclasses and attach them to self."""
        c_list = CPlot.get_subclasses()  # Retrieve all CPlot subclasses
        methods_list = []
        for c_info in c_list:  # For each CPlot subclass (name, class)
            if c_info[0] == CPlot.__name__:
                # Avoid adding methods of CPlot to CPlot
                continue
            # Get methods of each CPlot subclasses,
            # use isfunction for Py3, ismethod for Py2  # TODO: REMOVE Python 2
            pred = inspect.isfunction  # unbound methods or functions
            if sys.version_info < (3, 0):  # Py2 this covers unbound methods
                pred = inspect.ismethod
            c_methods = inspect.getmembers(c_info[1], pred)
            for method in c_methods:  # For each method (name, unbound method)
                # Skip special methods and already added methods
                if not method[0].startswith('__') and \
                        method[0] not in methods_list and \
                        not hasattr(self, method[0]):
                    methods_list.append(method)
        # Add methods to CPlot. Use __get__ to bound method to CPlot instance
        for method in methods_list:
            setattr(self, method[0], method[1].__get__(self))

    @property
    def n_lines(self):
        """Returns the number of lines inside current subplot."""
        return len(self.get_lines())

    def _set_lines_params(self, kwargs):
        """Add lines-related parameters to input dictionary."""
        # Parameters are updated/added only if not yet specified
        if 'linewidth' not in kwargs:
            kwargs['linewidth'] = self._params['lines.linewidth']
        if 'markersize' not in kwargs:
            kwargs['markersize'] = self._params['lines.markersize']

        return kwargs

    def set(self, param_name, param_value, copy=False):
        raise NotImplementedError

    def get_params(self):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def set_state(self, state_dict, copy=False):
        raise NotImplementedError

    def load_state(self, path):
        raise NotImplementedError

    def save_state(self, path):
        raise NotImplementedError

    def get_lines(self):
        """Return a list of lines contained by the subplot."""
        return self._sp.get_lines()

    def get_legend_handles_labels(self):
        """Return handles and labels for legend contained by the subplot."""
        return self._sp.get_legend_handles_labels()

    def get_xticks_idx(self, xticks):
        """Returns the position of markers to plot.

        Parameters
        ----------
        xticks : CArray
            Ticks of x-axis where marker should be plotted.

        Returns
        -------
        ticks_idx : list
            List with the position of each xtick.

        Notes
        -----
        If a given xtick is not exactly available,
        the closest value's position will be returned.

        """
        return xticks.binary_search(self._sp.get_xticks()).tolist()

    def set_axisbelow(self, axisbelow=True):
        """Set axis ticks and gridlines below most artists."""
        self._sp.set_axisbelow(axisbelow)

    def merge(self, sp):
        """Merge input subplot to active subplot.

        Parameters
        ----------
        sp : CPlot
            Subplot to be merged.

        """
        for line in sp.get_lines():
            self._sp.add_line(line)
        if self.get_legend() is not None:
            h, l = sp.get_legend_handles_labels()
            self.legend(h, l)

    def plot(self, x, y=None, *args, **kwargs):
        """Plot a line.

        If only one array is given it is supposed to be the
        y axis data. x axis values are set as index array 0..N-1 .

        Parameters
        ----------
        x : list or CArray
            x axis values
        y : list or CArray
            y axis values
        color : str

            .. list-table::
              :header-rows: 1

              * - Character
                - Color
              * - 'b'
                - blue
              * - 'g'
                - green
              * - 'r'
                - red
              * - 'c'
                - cyan
              * - 'm'
                - magenta
              * - 'y'
                - yellow
              * - 'k'
                - black
              * - 'w'
                - white
        alpha : float, default 1.0
            0.0 for transparent through 1.0 opaque
        linestyle : character, default '-'
            Can be one into this list : ['-' | '--' | '-.' | ':' | 'None' | ' ' | '']
        linewidth : float
            0.0 to 1.0
        marker : str

            .. list-table::
              :header-rows: 1

              * - Character
                - Marker
              * - '.'
                - point marker
              * - ','
                - pixel marker
              * - 'o'
                - circle marker
              * - 'v'
                - triangle_down marker
              * - '^'
                - triangle_up marker
              * - '<'
                - triangle_left marker
              * -  '>'
                - triangle_right marker
              * -  '1'
                - tri_down marker
              * - '2'
                - tri_up marker
              * - '3'
                - tri_left marker
              * - '4'
                - tri_right marker
              * - 's'
                - square marker
              * -  'p'
                - pentagon marker
              * -  '*'
                - star marker
              * -  'h'
                - hexagon1 marker
              * -  'H'
                - hexagon2 marker
              * -  '+'
                - plus marker
              * -  'x'
                - x marker
              * -  'D'
                - diamond marker
              * -  'd'
                - thin_diamond marker
              * - '|'
                - vline marker
              * - '_'
                - hline marker

        Examples
        --------
        .. plot:: pyplots/plot.py
            :include-source:

        """
        # Set lines-related parameters
        kwargs = self._set_lines_params(kwargs)
        # Convert sequences inside tuple to ndarray
        x, y = tuple_sequence_tondarray((x, y))
        if y is None:
            self._sp.plot(x, *args, **kwargs)
        else:
            self._sp.plot(x, y, *args, **kwargs)

    def semilogx(self, x, y=None, *args, **kwargs):
        """Plot with log scaling on the x axis.

        If only one array is given it is supposed to be the
        y axis data. x axis values are set as index array 0..N-1 .

        Parameters
        ----------
        x : list or CArray
            x axis values
        y : list or CArray
            y axis values
        basex : scalar > 1, default is 10
            Base of the x logarithm
        subsx : [ None | sequence ]
            Where to place the subticks between each major tick.
            Sequence of integers. For example, in a log10 scale:
            [2, 3, 4, 5, 6, 7, 8, 9] will place 8 logarithmically
            spaced minor ticks between each major tick.
        nonposx : [ 'mask' | 'clip' ], default 'mask'
            Non-positive values in x can be masked as invalid, or
            clipped to a very small positive number

        See Also
        --------
        .plot : Plot with standard axis.

        Examples
        --------
        .. plot:: pyplots/semilogx.py
            :include-source:

        """
        if 'subsx' in kwargs and isinstance(kwargs['subsx'], CArray):
                kwargs['subsx'] = kwargs['subsx'].tondarray()
        # Set other lines-related parameters
        kwargs = self._set_lines_params(kwargs)
        # Convert sequences inside tuple to ndarray
        x, y = tuple_sequence_tondarray((x, y))
        if y is None:
            self._sp.semilogx(x, *args, **kwargs)
        else:
            self._sp.semilogx(x, y, *args, **kwargs)

    def semilogy(self, x, y=None, *args, **kwargs):
        """Plot with log scaling on the y axis.

        If only one array is given it is supposed to be the
        y axis data. x axis values are set as index array 0..N-1 .

        Parameters
        ----------
        x : list or CArray
            x axis values.
        y : list or CArray
            y axis values.
        basey : scalar > 1, default is 10
            Base of the y logarithm
        subsy : [ None | sequence ], default None
            Where to place the subticks between each major tick.
            Should be a sequence of integers.
            For example, in a log10 scale: [2, 3, 4, 5, 6, 7, 8, 9]
            will place 8 logarithmically spaced minor ticks between
            each major tick.
        nonposy : [ 'mask' | 'clip' ], default 'mask'
            Non-positive values in x can be masked as invalid, or
            clipped to a very small positive number.

        See Also
        --------
        .plot : Plot with standard axis.

        Examples
        --------
        .. plot:: pyplots/semilogy.py
            :include-source:

        """
        if 'subsy' in kwargs and isinstance(kwargs['subsy'], CArray):
                kwargs['subsy'] = kwargs['subsy'].tondarray()
        # Set other lines-related parameters
        kwargs = self._set_lines_params(kwargs)
        # Convert sequences inside tuple to ndarray
        x, y = tuple_sequence_tondarray((x, y))
        if y is None:
            self._sp.semilogy(x, *args, **kwargs)
        else:
            self._sp.semilogy(x, y, *args, **kwargs)

    def loglog(self, x, y=None, *args, **kwargs):
        """Plot with log scaling on both the x and y axis.

        If only one array is given it is supposed to be the
        y axis data. x axis values are set as index array 0..N-1 .

        Parameters
        ----------
        x : list or CArray
            x axis values.
        y : list or CArray
            y axis values.
        basex, basey : scalar  > 1, default is 10
            Base of the x/y logarithm.
        subsx, subsy : [ None | sequence ]
            Where to place the subticks between each major tick.
            Should be a sequence of integers. For example, in a
            log10 scale: [2, 3, 4, 5, 6, 7, 8, 9] will place 8
            logarithmically spaced minor ticks between each major tick.
        nonposx, nonposy : ['mask' | 'clip' ], default 'mask'.
            Non-positive values in x or y can be masked as invalid, or
            clipped to a very small positive number.

        See Also
        --------
        .plot : Plot with standard axis.

        """
        if 'subsx' in kwargs and isinstance(kwargs['subsx'], CArray):
                kwargs['subsx'] = kwargs['subsx'].tondarray()
        if 'subsy' in kwargs and isinstance(kwargs['subsy'], CArray):
                kwargs['subsy'] = kwargs['subsy'].tondarray()
        # Set other lines-related parameters
        kwargs = self._set_lines_params(kwargs)
        # Convert sequences inside tuple to ndarray
        x, y = tuple_sequence_tondarray((x, y))
        if y is None:
            self._sp.loglog(x, *args, **kwargs)
        else:
            self._sp.loglog(x, y, *args, **kwargs)

    def scatter(self, x, y, s=20, c='b', *args, **kwargs):
        """Scatter plot of x vs y.

        Parameters
        ----------
        x, y : list or CArray
            Input data. Both object must have the same size.
        s : scalar or shape (n, ), optional, default: 20
            size in points^2.
        c : color or sequence of color, optional, default 'b'
            c can be a single color format string, or a sequence
            of color specifications of length N, or a sequence of
            numbers with the same shape of x,y to be mapped to
            colors using the cmap and norm specified via kwargs
            (see below). Note that c should not be a single
            numeric RGB or RGBA sequence because that is
            indistinguishable from an array of values to be
            colormapped. c can be a 2-D array in which the rows
            are RGB or RGBA, however.
        marker : MarkerStyle, optional, default: 'o'
            See markers for more information on the different
            styles of markers scatter supports.
        cmap : Colormap, optional, default: None
            A Colormap instance or registered name. cmap is only
            used if c is an array of floats. If None, default
            parameter image.cmap is used.
        norm : Normalize, optional, default: None
            A Normalize instance is used to scale luminance data
            to 0, 1. norm is only used if c is an array of floats.
        vmin, vmax : scalar, optional, default: None
            vmin and vmax are used in conjunction with norm to
            normalize luminance data. If either are None, the min
            and max of the color array is used. Note if you pass a
            norm instance, your settings for vmin and vmax will
            be ignored.
        alpha : scalar, optional, default: None
            The alpha blending value, between 0 (transparent) and 1 (opaque)
        linewidths : scalar or array_like, optional, default: None
            If None, defaults to (lines.linewidth,). Note that this
            is a tuple, and if you set the linewidths argument you
            must set it as a sequence of float.

        Examples
        --------
        .. plot:: pyplots/scatter.py
            :include-source:

        """
        if 'linewidths' not in kwargs:
            kwargs['linewidths'] = self._params['lines.linewidth']
        # Convert sequences inside tuple to ndarray
        if not isinstance(c, str):
            x, y, c = tuple_sequence_tondarray((x, y, c))
        else:
            x, y = tuple_sequence_tondarray((x, y))
        self._sp.scatter(x, y, s, c, *args, **kwargs)

    def contour(self, x, y, z, *args, **kwargs):
        """Draw contour lines of a function.

        Parameters
        ----------
        x, y : CArray or list
            specify the (x, y) coordinates of the surface.
            X and Y must both be 2-D with the same shape
            as Z, or they must both be 1-D such that len(X)
            is the number of columns in Z and len(Y) is the
            number of rows in Z.
        z : CArray or list
            value into (x, y) surface's position
        colors : [ None | string | (mpl_colors) ]
            If None, the colormap specified by cmap will be used.
            If a string, like 'r' or 'red', all levels will
            be plotted in this color.
            If a tuple of matplotlib color args (string, float,
            rgb, etc), different levels will be plotted in
            different colors in the order specified.
        alpha : float
            The alpha blending value
        cmap : [ None | Colormap ]
            A cm Colormap instance or None. If cmap is None and
            colors is None, a default Colormap is used.
        vmin, vmax : [ None | scalar ]
            If not None, either or both of these values will be
            supplied to the matplotlib.colors.
            Normalize instance, overriding the default color
            scaling based on levels.
        levels : [level0, level1, ..., leveln]
            A list of floating point numbers indicating the level
            curves to draw; e.g., to draw just the zero contour
            pass levels=[0]
        origin : [ None | 'upper' | 'lower' | 'image' ]
            If None, the first value of Z will correspond to the
            lower left corner, location (0,0). If 'image', the
            default parameter value for image.origin will be used.
            This keyword is not active if X and Y are specified
            in the call to contour.
        extent : [ None | (x0,x1,y0,y1) ]
            If origin is not None, then extent is interpreted as
            in matplotlib.pyplot.imshow(): it gives the outer
            pixel boundaries. In this case, the position of Z[0,0]
            is the center of the pixel, not a corner.
            If origin is None, then (x0, y0) is the position of Z[0,0],
            and (x1, y1) is the position of Z[-1,-1].
            This keyword is not active if X and Y are specified in
            the call to contour.
        extend : [ 'neither' | 'both' | 'min' | 'max' ]
            Unless this is 'neither', contour levels are automatically
            added to one or both ends of the range so that all data are included.
            These added ranges are then mapped to the special colormap
            values which default to the ends of the colormap range.
        antialiased : [ True | False ]
            enable antialiasing, overriding the defaults.
            For filled contours, the default is True. For line
            contours, it is taken from default_parameters ['lines.antialiased'].
        linewidths : [ None | number | tuple of numbers ]
            If linewidths is None, the default width in lines.linewidth
            default_parameters is used.
            If a number, all levels will be plotted with this linewidth.
            If a tuple, different levels will be plotted with different
            linewidths in the order specified.
        linestyles : [ None | 'solid' | 'dashed' | 'dashdot' | 'dotted' ]
            If linestyles is None, the default is 'solid' unless the
            lines are monochrome. In that case, negative contours will
            take their linestyle from the matplotlibrc `contour.negative_`
            linestyle setting.
            linestyles can also be an iterable of the above strings
            specifying a set of linestyles to be used. If this iterable
            is shorter than the number of contour levels it will be
            repeated as necessary.

        Examples
        --------
        .. plot:: pyplots/contour.py
            :include-source:

        """
        if 'linewidths' not in kwargs:
            kwargs['linewidths'] = self._params['lines.linewidth']
        # Convert sequences inside tuple to ndarray
        x, y, z = tuple_sequence_tondarray((x, y, z))
        return self._sp.contour(x, y, z, *args, **kwargs)

    def contourf(self, x, y, z, *args, **kwargs):
        """Draw filled contour of a function.

        Parameters
        ----------
        x, y : CArray or list
            specify the (x, y) coordinates of the surface.
            X and Y must both be 2-D with the same shape as Z,
            or they must both be 1-D such that len(X) is the
            number of columns in Z and len(Y) is the number of
            rows in Z.
        z : CArray or list
            value into (x, y) surface's position
        colors : [ None | string | (mpl_colors) ]
            If None, the colormap specified by cmap will be used.
            If a string, like 'r' or 'red', all levels will be
            plotted in this color.
            If a tuple of matplotlib color args (string, float,
            rgb, etc), different levels will be plotted in different
            colors in the order specified.
        alpha : float
            The alpha blending value
        cmap : [ None | Colormap ]
            A cm Colormap instance or None. If cmap is None and
            colors is None, a default Colormap is used.
        vmin, vmax : [ None | scalar ]
            If not None, either or both of these values will be
            supplied to the matplotlib.colors.
            Normalize instance, overriding the default color
            scaling based on levels.
        levels : [level0, level1, ..., leveln]
            A list of floating point numbers indicating the level
            curves to draw; e.g., to draw just the zero contour
            pass levels=[0]
        origin : [ None | 'upper' | 'lower' | 'image' ]
            If None, the first value of Z will correspond to the
            lower left corner, location (0,0). If 'image', the
            default parameter value for image.origin will be used.
            This keyword is not active if X and Y are specified
            in the call to contour.
        extent : [ None | (x0,x1,y0,y1) ]
            If origin is not None, then extent is interpreted as
            in matplotlib.pyplot.imshow(): it gives the outer
            pixel boundaries.
            In this case, the position of Z[0,0] is the center
            of the pixel, not a corner.
            If origin is None, then (x0, y0) is the position of
            Z[0,0], and (x1, y1) is the position of Z[-1,-1].
            This keyword is not active if X and Y are specified
            in the call to contour.
        extend : [ 'neither' | 'both' | 'min' | 'max' ]
            Unless this is 'neither', contour levels are automatically
            added to one or both ends of the range so that all data
            are included.
            These added ranges are then mapped to the special colormap
            values which default to the ends of the colormap range.
        antialiased : [ True | False ]
            enable antialiasing, overriding the defaults.
            For filled contours, the default is True. For line contours,
            it is taken from default_parameters ['lines.antialiased'].

        Examples
        --------
        .. plot:: pyplots/contourf.py
            :include-source:

        """
        # Convert sequences inside tuple to ndarray
        x, y, z = tuple_sequence_tondarray((x, y, z))
        return self._sp.contourf(x, y, z, *args, **kwargs)

    def clabel(self, contour, *args, **kwargs):
        """Label a contour plot.

        Parameters
        ----------
        contour : contour object
            returned from contour function
        fontsize : int
            size in points or relative size e.g., 'smaller', 'x-large'
        colors : str
            if None, the color of each label matches the color
            of the corresponding contour
            if one string color, e.g., colors = 'r' or
            colors = 'red', all labels will be plotted in
            this color
            if a tuple of matplotlib color args (string, float,
            rgb, etc), different labels will be plotted in different
            colors in the order specified
        inline : bool
            controls whether the underlying contour is removed
            or not. Default is True.
        inline_spacing : int
            space in pixels to leave on each side of label when
            placing inline. Defaults to 5.
            This spacing will be exact for labels at locations
            where the contour is straight, less so for labels on
            curved contours.
        fmt : str
            a format string for the label. Default is '%1.3f'
            Alternatively, this can be a dictionary matching contour
            levels with arbitrary strings to use for each contour
            level (i.e., fmt[level]=string), or it can be any callable,
            such as a Formatter instance, that returns a string when
            called with a numeric contour level.
        manual : bool
            if True, contour labels will be placed manually using
            mouse clicks.
            Click the first button near a contour to add a label,
            click the second button (or potentially both mouse
            buttons at once) to finish adding labels. The third
            button can be used to remove the last label added, but
            only if labels are not inline. Alternatively, the
            keyboard can be used to select label locations (enter
            to end label placement, delete or backspace act like
            the third mouse button, and any other key will select
            a label location).
            manual can be an iterable object of x,y tuples. Contour
            labels will be created as if mouse is clicked at each
            x,y positions.
        rightside_up : bool
            if True (default), label rotations will always be plus
            or minus 90 degrees from level.

        Examples
        --------
        .. plot:: pyplots/clabel.py
            :include-source:

        """
        if 'fontsize' not in kwargs:
            kwargs['fontsize'] = self._params['font.size']
        return self._sp.clabel(contour, *args, **kwargs)

    def colorbar(self, mappable, ticks=None, *args, **kwargs):
        """Add colorbar to plot.

        Parameters
        ----------
        mappable : object
            Image, ContourSet, or other to which the colorbar applies
        use_gridspec : boolean, default False
            if True colorbar is created as an instance of Subplot using the grid_spec module.

        Additional keyword arguments are of two kinds:

        Axes properties:

            .. list-table::
              :header-rows: 1

              * - Property
                - Description
              * - orientation
                - vertical or horizontal
              * - fraction, default 0.15
                - fraction of original axes to use for colorbar
              * - pad, default 0.05 if vertical, 0.15 if horizontal
                - fraction of original axes between colorbar and new image axes
              * - shrink, default 1.0
                - fraction by which to shrink the colorbar
              * - aspect, default 20
                - ratio of long to short dimensions
              * - anchor, default (0.0, 0.5) if vertical; (0.5, 1.0) if horizontal
                - the anchor point of the colorbar axes
              * - panchor, default (1.0, 0.5) if vertical; (0.5, 0.0) if horizontal;
                - the anchor point of the colorbar parent axes. If False, the
                  parent axes' anchor will be unchanged

        Colorbar properties:

            .. list-table::
              :header-rows: 1

              * - Property
                - Description
              * - extend
                - [ 'neither' | 'both' | 'min' | 'max' ] If not 'neither', make
                  pointed end(s) for out-of- range values. These are set for a
                  given colormap using the colormap set_under and set_over methods.
              * - extendfrac
                - [ None | 'auto' | length | lengths ] If set to None, both the
                  minimum and maximum triangular colorbar extensions with have a
                  length of 5% of the interior colorbar length (this is the default
                  setting). If set to 'auto', makes the triangular colorbar
                  extensions the same lengths as the interior boxes (when spacing
                  is set to 'uniform') or the same lengths as the respective adjacent
                  interior boxes (when spacing is set to 'proportional'). If a scalar,
                  indicates the length of both the minimum and maximum triangular colorbar
                  extensions as a fraction of the interior colorbar length. A two-element
                  sequence of fractions may also be given, indicating the lengths of the
                  minimum and maximum colorbar extensions respectively as a fraction of
                  the interior colorbar length.
              * - extendrect
                - [ False | True ] If False the minimum and maximum colorbar extensions
                  will be triangular (the default). If True the extensions will be rectangular.
              * - spacing
                - [ 'uniform' | 'proportional' ] Uniform spacing gives each discrete color
                  the same space; proportional makes the space proportional to the data interval.
              * - ticks
                - [ None | list of ticks | Locator object ] If None, ticks are determined
                  automatically from the input.
              * - format
                - [ None | format string | Formatter object ] If None, the ScalarFormatter
                  is used. If a format string is given, e.g., '%.3f', that is used. An
                  alternative Formatter object may be given instead.
              * - drawedges
                - [ False | True ] If true, draw lines at color boundaries.

        Notes
        -----
        If mappable is a ContourSet, its extend kwarg is included automatically.
        Note that the shrink kwarg provides a simple way to keep a vertical colorbar.
        If the colorbar is too tall (or a horizontal colorbar is too wide) use a
        smaller value of shrink.

        Examples
        --------
        .. plot:: pyplots/colorbar.py
            :include-source:

        """
        ticks = ticks.tolist() if isinstance(ticks, CArray) else ticks
        from matplotlib.pyplot import colorbar
        cbar = colorbar(mappable, ticks=ticks, *args, **kwargs)
        if 'fontsize' not in kwargs:
            kwargs['fontsize'] = self._params['font.size']
        cbar.ax.tick_params(labelsize=kwargs['fontsize'])
        return cbar

    def errorbar(self, x, y, xerr=None, yerr=None, *args, **kwargs):
        """Plot with error deltas in yerr and xerr.

        Vertical errorbars are plotted if yerr is not None. Horizontal
        errorbars are plotted if xerr is not None. x, y, xerr, and yerr
        can all be scalars, which plots a single error bar at x, y.

        Parameters
        ----------
        x : list or CArray
            x axis values.
        y : list or CArray
            y axis values.
        xerr, yerr : [ scalar | N, Nx1, or 2xN array-like ], default None
            If a scalar number, len(N) array-like object, or an
            Nx1 array-like object, errorbars are drawn at +/-value
            relative to the data.
            If a sequence of shape 2xN, errorbars are drawn at -row1
            and +row2 relative to the data.
        fmt : [ '' | 'none' | plot format string ], default ''
            The plot format symbol. If fmt is 'none' (case-insensitive),
            only the errorbars are plotted.
            This is used for adding errorbars to a bar plot, for example.
            Default is '', an empty plot format string; properties are
            then identical to the defaults for plot().
        ecolor : [ None | mpl color ], default None
            A matplotlib color arg which gives the color the errorbar
            lines; if None, use the color of the line connecting the markers.
        elinewidth : scalar, default None
            The linewidth of the errorbar lines. If None, use the linewidth.
        capsize : scalar, default 3
            The length of the error bar caps in points.
        capthick : scalar, default None
            An alias kwarg to markeredgewidth (a.k.a. - mew). This setting
            is a more sensible name for the property that controls the
            thickness of the error bar cap in points. For backwards
            compatibility, if mew or markeredgewidth are given, then they
            will over-ride capthick. This may change in future releases.
        barsabove : [ True | False ]
            if True, will plot the errorbars above the plot
            symbols. Default is below.
        lolims, uplims, xlolims, xuplims : [ False | True ], default False
            These arguments can be used to indicate that a value
            gives only upper/lower limits. In that case a caret symbol
            is used to indicate this. lims-arguments may be of the same
            type as xerr and yerr. To use limits with inverted axes,
            set_xlim() or set_ylim() must be called before errorbar().
        errorevery : positive integer, default 1
            subsamples the errorbars. e.g., if everyerror=5, errorbars
            for every 5-th datapoint will be plotted. The data plot
            itself still shows all data points.

        Examples
        --------
        .. plot:: pyplots/errorbar.py
            :include-source:

        """
        # Set lines-related parameters
        kwargs = self._set_lines_params(kwargs)
        # Convert sequences inside tuple to ndarray
        x, y, xerr, yerr = tuple_sequence_tondarray((x, y, xerr, yerr))
        self._sp.errorbar(x, y, xerr=xerr, yerr=yerr, *args, **kwargs)

    def bar(self, left, height, width=0.8, bottom=None, *args, **kwargs):
        """Bar plot.

        Parameters
        ----------
        left : sequence of scalars
            x coordinates of the left sides of the bars.
        height : sequence of scalars
            height(s) of the bars.
        width : scalar or array-like, optional, default: 0.8
            width(s) of the bars.
        bottom : scalar or array-like, optional, default: None
            y coordinate(s) of the bars.
        color : scalar or array-like, optional
            Colors of the bar faces.
        edgecolor : scalar or array-like, optional
            Colors of the bar edges.
        linewidth : scalar or array-like, optional, default: None
            Width of bar edge(s). If None, use default linewidth;
            If 0, don't draw edges.
        xerr : scalar or array-like, optional, default: None
            If not None, will be used to generate errorbar(s)
            on the bar chart.
        yerr : scalar or array-like, optional, default: None
            If not None, will be used to generate errorbar(s)
            on the bar chart.
        ecolor : scalar or array-like, optional, default: None
            Specifies the color of errorbar(s)
        capsize : integer, optional, default: 3
            Determines the length in points of the error bar caps.
        error_kw : dict
            dictionary of kwargs to be passed to errorbar method.
            ecolor and capsize may be specified here rather than
            independent kwargs.
        align : ['edge' | 'center'], optional, default: 'edge'
            If edge, aligns bars by their left edges (for vertical
            bars) and by their bottom edges (for horizontal bars).
            If center, interpret the left argument as the coordinates
            of the centers of the bars.
        orientation : 'vertical' | 'horizontal', optional, default: 'vertical'
            The orientation of the bars.
        log : boolean, optional, default: False
            If true, sets the axis to be log scale.

        Returns
        -------
            bar_list : list of bar type objects

        Examples
        --------
        .. plot:: pyplots/bar.py
            :include-source:

        """
        if 'linewidth' not in kwargs:
            kwargs['linewidth'] = self._params['lines.linewidth']
        # Convert sequences inside tuple to ndarray
        left, height, width, bottom = tuple_sequence_tondarray(
            (left, height, width, bottom))
        return self._sp.bar(left, height, width, bottom, *args, **kwargs)

    def barh(self, bottom, width, height=0.8, left=None, *args, **kwargs):
        """Horizontal bar plot.

        Parameters
        ----------
        bottom : sequence of scalars
            y coordinates of the bars.
        width : sequence of scalars
            width(s) of the bars.
        height : scalar or array-like, optional, default: 0.8
            height(s) of the bars.
        left : scalar or array-like, optional, default: None
            x coordinate(s) of the bars.
        color : scalar or array-like, optional
            Colors of the bar faces.
        edgecolor : scalar or array-like, optional
            Colors of the bar edges.
        linewidth : scalar or array-like, optional, default: None
            Width of bar edge(s). If None, use default linewidth;
            If 0, don't draw edges.
        xerr : scalar or array-like, optional, default: None
            If not None, will be used to generate errorbar(s)
            on the bar chart.
        yerr : scalar or array-like, optional, default: None
            If not None, will be used to generate errorbar(s)
            on the bar chart.
        ecolor : scalar or array-like, optional, default: None
            Specifies the color of errorbar(s)
        capsize : integer, optional, default: 3
            Determines the length in points of the error bar caps.
        error_kw : dict
            dictionary of kwargs to be passed to errorbar method.
            ecolor and capsize may be specified here rather than
            independent kwargs.
        align : ['edge' | 'center'], optional, default: 'edge'
            If edge, aligns bars by their left edges (for vertical
            bars) and by their bottom edges (for horizontal bars).
            If center, interpret the left argument as the coordinates
            of the centers of the bars.
        orientation : 'vertical' | 'horizontal', optional, default: 'vertical'
            The orientation of the bars.
        log : boolean, optional, default: False
            If true, sets the axis to be log scale.

        Returns
        -------
        bar_list : list of bar type objects

        """
        if 'linewidth' not in kwargs:
            kwargs['linewidth'] = self._params['lines.linewidth']
        # Convert sequences inside tuple to ndarray
        bottom, width, height, left = tuple_sequence_tondarray(
            (bottom, width, height, left))
        return self._sp.barh(bottom, width, height, left, *args, **kwargs)

    def hist(self, x, *args, **kwargs):
        """Plot a histogram.

        Compute and draw the histogram of x.

        The return value is a tuple (n, bins, patches) or
        ([n0, n1, ...], bins, [patches0, patches1,...]) if
        the input contains multiple data.

        Multiple data can be provided via x as a list of
        datasets of potentially different length ([x0, x1, ...]),
        or as a 2-D ndarray in which each column is a dataset.

        Parameters
        ----------
        x : (n,) array or sequence of (n,) arrays
            Input values, this takes either a single array or a
            sequency of arrays which are not required to be of
            the same length
        bins : integer or array_like, optional, default is 10
            If an integer is given, bins + 1 bin edges are returned.
            Unequally spaced bins are supported if bins is a sequence.
        range : tuple or None, optional
            The lower and upper range of the bins. Lower and upper
            outliers are ignored.
            If not provided, range is (x.min(), x.max()). Range has
            no effect if bins is a sequence.
            If bins is a sequence or range is specified, autoscaling
            is based on the specified bin range instead of the range of x.
        density : boolean, optional
            If True, the first element of the return tuple will be
            the counts normalized to form a probability density,
            i.e., n/(len(x)`dbin), i.e., the integral of the histogram
            will sum to 1.  If stacked is also True, the sum of the
            histograms is normalized to 1.
        weights : (n, ) array_like or None, optional
            An array of weights, of the same shape as x. Each value
            in x only contributes its associated weight towards the
            bin count (instead of 1). If density is True, the weights
            are normalized, so that the integral of the density over
            the range remains 1.
        cumulative : boolean, optional
            Dafault False. If True, then a histogram is computed
            where each bin gives the counts in that bin plus all bins
            for smaller values.
            The last bin gives the total number of datapoints.
            If density is also True then the histogram is normalized
            such that the last bin equals 1.
            If cumulative evaluates to less than 0 (e.g., -1), the
            direction of accumulation is reversed. In this case, if density
            is also True, then the histogram is normalized such that
            the first bin equals 1.
        bottom : array_like, scalar, or None
            Location of the bottom baseline of each bin. If a scalar,
            the base line for each bin is shifted by the same amount.
            If an array, each bin is shifted independently and the
            length of bottom must match the number of bins.
            If None, defaults to 0.
        histtype : {'bar', 'barstacked', 'step', 'stepfilled'}, optional
            - 'bar' (default) is a traditional bar-type histogram.
              If multiple data are given the bars are aranged side by side.
            - 'barstacked' is a bar-type histogram where multiple data are
              stacked on top of each other.
            - 'step' generates a lineplot that is by default unfilled.
            - 'stepfilled' generates a lineplot that is by default filled.
        align : {'left', 'mid', 'right'}, optional
            - 'left': bars are centered on the left bin edges.
            - 'mid': default, bars are centered between the bin edges.
            - 'right': bars are centered on the right bin edges.
        orientation : {'horizontal', 'vertical'}, optional
            If 'horizontal', barh will be used for bar-type histograms
            and the bottom kwarg will be the left edges.
        rwidth : scalar or None, optional
            The relative width of the bars as a fraction of the bin width.
            If None, automatically compute the width. Ignored if histtype
            is 'step' or 'stepfilled'.
        log : boolean, optional
            Default False. If True, the histogram axis will be set to a
            log scale. If log is True and x is a 1D array, empty bins
            will be filtered out and only the non-empty (n, bins, patches)
            will be returned.
        color : color or array_like of colors or None, optional
            Color spec or sequence of color specs, one per dataset.
            Default (None) uses the standard line color sequence.
        label : string or None, optional
            String, or sequence of strings to match multiple datasets.
            Bar charts yield multiple patches per dataset, but only the
            first gets the label, so that the legend command will work
            as expected.
        stacked : boolean, optional
            If True, multiple data are stacked on top of each other.
            If False (default) multiple data are aranged side by side
            if histtype is 'bar' or on top of each other if histtype
            is 'step'.

        Returns
        -------
        n : CArray or list of arrays
            The values of the histogram bins. See density and weights
            for a description of the possible semantics.
            If input x is an array, then this is an array of length nbins.
            If input is a sequence arrays [data1, data2,..], then this is
            a list of arrays with the values of the histograms for each of
            the arrays in the same order.
        bins : CArray
            The edges of the bins. Length nbins + 1 (nbins left edges and
            right edge of last bin).
            Always a single array even when multiple data sets are passed in.
        patches : list or list of lists
            Silent list of individual patches used to create the histogram or
            list of such list if multiple input datasets.

        Examples
        --------
        .. plot:: pyplots/hist.py
            :include-source:

        """
        if 'linewidth' not in kwargs:
            kwargs['linewidth'] = self._params['lines.linewidth']
        x = list(xi.tondarray() if isinstance(xi, CArray) else xi for xi in x)
        n, bins, patches = self._sp.hist(x, *args, **kwargs)
        if isinstance(n, list):
            n = list(CArray(ni) for ni in n)
        return n, CArray(bins), patches

    def boxplot(self, x, notch=False, sym=None, vert=True, whis=1.5,
                positions=None, widths=None, patch_artist=False,
                bootstrap=None, usermedians=None, conf_intervals=None,
                meanline=False, showmeans=False, showcaps=True,
                showbox=True, showfliers=True, boxprops=None, labels=None,
                flierprops=None, medianprops=None, meanprops=None,
                capprops=None, whiskerprops=None, manage_xticks=True):
        """Make a box and whisker plot.

        Make a box and whisker plot for each column of *x* or each
        vector in sequence *x*.  The box extends from the lower to
        upper quartile values of the data, with a line at the median.
        The whiskers extend from the box to show the range of the
        data.  Flier points are those past the end of the whiskers.

        Parameters
        ----------
        x : Array or a sequence of vectors.
            The input data.
        notch : bool, default = False
            If False, produces a rectangular box plot.
            If True, will produce a notched box plot
        sym : str or None, default = None
            The default symbol for flier points.
            Enter an empty string ('') if you don't want to show fliers.
            If `None`, then the fliers default to 'b+'  If you want more
            control use the flierprops kwarg.
        vert : bool, default = True
            If True (default), makes the boxes vertical.
            If False, makes horizontal boxes.
        whis : float, sequence (default = 1.5) or string
            As a float, determines the reach of the whiskers past the first
            and third quartiles (e.g., Q3 + whis*IQR, IQR = interquartile
            range, Q3-Q1). Beyond the whiskers, data are considered outliers
            and are plotted as individual points. Set this to an unreasonably
            high value to force the whiskers to show the min and max values.
            Alternatively, set this to an ascending sequence of percentile
            (e.g., [5, 95]) to set the whiskers at specific percentiles of
            the data. Finally, *whis* can be the string 'range' to force the
            whiskers to the min and max of the data. In the edge case that
            the 25th and 75th percentiles are equivalent, *whis* will be
            automatically set to 'range'.
        bootstrap : None (default) or integer
            Specifies whether to bootstrap the confidence intervals
            around the median for notched boxplots. If bootstrap==None,
            no bootstrapping is performed, and notches are calculated
            using a Gaussian-based asymptotic approximation  (see McGill, R.,
            Tukey, J.W., and Larsen, W.A., 1978, and Kendall and Stuart,
            1967). Otherwise, bootstrap specifies the number of times to
            bootstrap the median to determine it's 95% confidence intervals.
            Values between 1000 and 10000 are recommended.
        usermedians : array-like or None (default)
            An array or sequence whose first dimension (or length) is
            compatible with *x*. This overrides the medians computed by
            matplotlib for each element of *usermedians* that is not None.
            When an element of *usermedians* == None, the median will be
            computed by matplotlib as normal.
        conf_intervals : array-like or None (default)
            Array or sequence whose first dimension (or length) is compatible
            with *x* and whose second dimension is 2. When the current element
            of *conf_intervals* is not None, the notch locations computed by
            matplotlib are overridden (assuming notch is True). When an
            element of *conf_intervals* is None, boxplot compute notches the
            method specified by the other kwargs (e.g., *bootstrap*).
        positions : array-like, default = [1, 2, ..., n]
            Sets the positions of the boxes. The ticks and limits
            are automatically set to match the positions.
        widths : array-like, default = 0.5
            Either a scalar or a vector and sets the width of each box. The
            default is 0.5, or ``0.15*(distance between extreme positions)``
            if that is smaller.
        labels : sequence or None (default)
            Labels for each dataset. Length must be compatible with
            dimensions  of *x*
        patch_artist : bool, default = False
            If False produces boxes with the Line2D artist
            If True produces boxes with the Patch artist
        showmeans : bool, default = False
            If True, will toggle one the rendering of the means
        showcaps : bool, default = True
            If True, will toggle one the rendering of the caps
        showbox : bool, default = True
            If True, will toggle one the rendering of box
        showfliers : bool, default = True
            If True, will toggle one the rendering of the fliers
        boxprops : dict or None (default)
            If provided, will set the plotting style of the boxes
        whiskerprops : dict or None (default)
            If provided, will set the plotting style of the whiskers
        capprops : dict or None (default)
            If provided, will set the plotting style of the caps
        flierprops : dict or None (default)
            If provided, will set the plotting style of the fliers
        medianprops : dict or None (default)
            If provided, will set the plotting style of the medians
        meanprops : dict or None (default)
            If provided, will set the plotting style of the means
        meanline : bool, default = False
            If True (and *showmeans* is True), will try to render the mean
            as a line spanning the full width of the box according to
            *meanprops*. Not recommended if *shownotches* is also True.
            Otherwise, means will be shown as points.

        Returns
        -------
        result : dict
            A dictionary mapping each component of the boxplot
            to a list of the :class:`matplotlib.lines.Line2D`
            instances created. That dictionary has the following keys
            (assuming vertical boxplots):

            - boxes: the main body of the boxplot showing the quartiles
              and the median's confidence intervals if enabled.
            - medians: horizonal lines at the median of each box.
            - whiskers: the vertical lines extending to the most extreme,
              n-outlier data points.
            - caps: the horizontal lines at the ends of the whiskers.
            - fliers: points representing data that extend beyond the
              whiskers (outliers).
            - means: points or lines representing the means.

        """
        if isinstance(x, CArray):
            x = (x, )
        x = tuple_sequence_tondarray(tuple(x))
        if usermedians is not None:
            if isinstance(usermedians, CArray):
                usermedians = (usermedians, )
            usermedians = tuple_sequence_tondarray(tuple(usermedians))
        if conf_intervals is not None:
            if isinstance(conf_intervals, CArray):
                conf_intervals = (conf_intervals, )
            conf_intervals = tuple_sequence_tondarray(tuple(conf_intervals))
        if isinstance(positions, CArray):
            positions = positions.tondarray()

        self._sp.boxplot(x, notch, sym, vert, whis,
                         positions, widths, patch_artist,
                         bootstrap, usermedians, conf_intervals,
                         meanline, showmeans, showcaps,
                         showbox, showfliers, boxprops,
                         labels, flierprops, medianprops,
                         meanprops, capprops, whiskerprops,
                         manage_xticks)

    def fill_between(self, x, y1, y2=0, where=None,
                     interpolate=False, step=None, **kwargs):
        """Fill the area between two horizontal curves.

        The curves are defined by the points (x, y1) and (x, y2).
        This creates one or multiple polygons describing the filled area.

        You may exclude some horizontal sections from filling using where.

        By default, the edges connect the given points directly.
        Use step if the filling should be a step function,
        i.e. constant in between x.

        Parameters
        ----------
        x : CArray (length N)
            The x coordinates of the nodes defining the curves.
        y1 : CArray (length N) or scalar
            The y coordinates of the nodes defining the first curve.
        y2 : CArray (length N) or scalar, optional, default: 0
            The y coordinates of the nodes defining the second curve.
        where : CArray of bool (length N), optional, default: None
            Define where to exclude some horizontal regions from being filled.
            The filled regions are defined by the coordinates x[where].
            More precisely, fill between x[i] and x[i+1] if where[i] and
            where[i+1]. Note that this definition implies that an isolated
            True value between two False values in where will not result
            in filling. Both sides of the True position remain unfilled due
            to the adjacent False values.
        interpolate : bool, optional
            This option is only relvant if where is used and the two curves
            are crossing each other.
            Semantically, where is often used for y1 > y2 or similar.
            By default, the nodes of the polygon defining the filled region
            will only be placed at the positions in the x array.
            Such a polygon cannot describe the above semantics close to
            the intersection. The x-sections containing the intersecion
            are simply clipped.
            Setting interpolate to True will calculate the actual
            intersection point and extend the filled region up to this point.
        step : {'pre', 'post', 'mid'}, optional
            Define step if the filling should be a step function,
            i.e. constant in between x.
            The value determines where the step will occur:

             - 'pre': The y value is continued constantly to the left from
               every x position, i.e. the interval (x[i-1], x[i]] has the
               value y[i].
             - 'post': The y value is continued constantly to the right from
               every x position, i.e. the interval [x[i], x[i+1]) has the
               value y[i].
             - 'mid': Steps occur half-way between the x positions.

        """
        x, y1, y2, where = tuple_sequence_tondarray((x, y1, y2, where))
        self._sp.fill_between(x, y1, y2=y2, where=where,
                              interpolate=interpolate, step=step, **kwargs)

    def xlim(self, bottom=None, top=None):
        """Set axes x limits.

        Parameters
        ----------
        bottom : scalar
            Starting value for the x axis.
        top : scalar
            Ending value for the x axis.

        Examples
        --------
        .. plot:: pyplots/xlim.py
            :include-source:

        """
        self._xlim = (bottom, top)
        self._sp.set_xlim(bottom, top)

    def ylim(self, bottom=None, top=None):
        """Set axes y limits.

        Parameters
        ----------
        bottom : scalar
            Starting value for the y axis.
        top : scalar
            Ending value for the y axis.

        See Also
        --------
        .xlim : Set x axis limits.

        """
        self._ylim = (bottom, top)
        self._sp.set_ylim(bottom, top)

    def xscale(self, scale_type, nonposx='mask', basex=10, **kwargs):
        """Set scale for x axis.

        Parameters
        ----------
        scale_type : {'linear', 'log', 'symlog', 'logit'}
            Scale for x axis. Default 'linear'.
        nonposx: [ 'mask' | 'clip' ], default 'mask'
            Non-positive values in x can be masked as invalid,
            or clipped to a very small positive number.
        basex : int
            The base of the logarithm, must be higger than 1.

        """
        self._sp.set_xscale(scale_type, nonposx=nonposx, basex=basex, **kwargs)

    def yscale(self, scale_type, nonposy='mask', basey=10, **kwargs):
        """Set scale for y axis.

        Parameters
        ----------
        scale_type : {'linear', 'log', 'symlog', 'logit'}
            Scale for y axis. Default 'linear'.
        nonposy: [ 'mask' | 'clip' ], default 'mask'
            Non-positive values in y can be masked as invalid,
            or clipped to a very small positive number.
        basey : int
            The base of the logarithm, must be higger than 1.

        """
        self._sp.set_yscale(scale_type, nonposy=nonposy, basey=basey, **kwargs)

    def xlabel(self, label, *args, **kwargs):
        """Set a label for the x axis.

        Parameters
        ----------
        label : string
            Label's text.
        *args, **kwargs
            Same as :meth:`.text` method.

        Examples
        --------
        .. plot:: pyplots/xlabel.py
            :include-source:

        """
        if 'fontsize' not in kwargs:
            kwargs['fontsize'] = self._params['font.size']
        self._xlabel = label
        self._sp.set_xlabel(label, *args, **kwargs)

    def ylabel(self, label, *args, **kwargs):
        """Set a label for the y axis

        Parameters
        ----------
        label : string
            Label's text.
        *args, **kwargs
            Same as :meth:`.text` method.

        See Also
        --------
        .xlabel : Set a label for the x axis.

        """
        if 'fontsize' not in kwargs:
            kwargs['fontsize'] = self._params['font.size']
        self._ylabel = label
        self._sp.set_ylabel(label, *args, **kwargs)

    def xticks(self, location_array, *args, **kwargs):
        """Set the x-tick locations and labels.

        Parameters
        ----------
        location_array : CArray or list
            Contain ticks location.
        *args, **kwargs
            Same as :meth:`.text` method.

        Examples
        --------
        .. plot:: pyplots/xticks.py
            :include-source:

        """
        if isinstance(location_array, CArray):
            location_array = location_array.tondarray()
        self._xticks = location_array
        self._sp.set_xticks(location_array, *args, **kwargs)

    def yticks(self, location_array, *args, **kwargs):
        """Set the y-tick locations and labels.

        Parameters
        ----------
        location_array : CArray or list
            Contain ticks location.
        *args, **kwargs
            Same as :meth:`.text` method.

        See Also
        --------
        .xticks : Set the x-tick locations and labels.

        """
        if isinstance(location_array, CArray):
            location_array = location_array.tondarray()
        self._yticks = location_array
        self._sp.set_yticks(location_array, *args, **kwargs)

    def xticklabels(self, labels, *args, **kwargs):
        """Set the xtick labels.

        Parameters
        ----------
        labels : list or CArray of string
            Xtick labels.
        *args, **kwargs
            Same as :meth:`.text` method.

        Examples
        --------
        .. plot:: pyplots/xticklabels.py
            :include-source:

        """
        labels = labels.tolist() if isinstance(labels, CArray) else labels
        self._xticklabels = labels
        self._sp.set_xticklabels(labels, *args, **kwargs)

    def yticklabels(self, labels, *args, **kwargs):
        """Set the ytick labels.

        Parameters
        ----------
        labels : list or CArray of string
            Xtick labels.
        *args, **kwargs
            Same as :meth:`.text` method.

        See Also
        --------
        .xticklabels : Set the xtick labels.

        """
        labels = labels.tolist() if isinstance(labels, CArray) else labels
        self._yticklabels = labels
        self._sp.set_yticklabels(labels, *args, **kwargs)

    def tick_params(self, *args, **kwargs):
        """Change the appearance of ticks and tick labels.

        Parameters
        ----------
        axis : ['x' | 'y' | 'both']
            Axis on which to operate; default is 'both'.
        reset : [True | False]
            Default False. If True, set all parameters to defaults
            before processing other keyword arguments.
        which : ['major' | 'minor' | 'both']
            Default is 'major'; apply arguments to which ticks.
        direction : ['in' | 'out' | 'inout']
            Puts ticks inside the axes, outside the axes, or both.
        length : int
            Tick length in points.
        width : int
            Tick width in points.
        color : str
            Tick color; accepts any mpl color spec.
        pad : int
            Distance in points between tick and label.
        labelsize : int, str
            Tick label font size in points or as a string (e.g., 'large').
        labelcolor : str
            Tick label color; mpl color spec.
        colors : str
            Changes the tick color and the label color to the same
            value: mpl color spec.
        bottom, top, left, right : bool, optional
            Controls whether to draw the respective ticks.
        labelbottom, labeltop, labelleft, labelright : bool, optional
            Controls whether to draw the respective tick labels.

        Examples
        --------
        .. plot:: pyplots/tick_params.py
            :include-source:

        """
        self._sp.tick_params(*args, **kwargs)

    def grid(self, grid_on=True, axis='both', **kwargs):
        """Draw grid for current plot.

        Parameters
        ----------
        grid_on : boolean, default True
            if True show grid, elsewhere hide grid.
        axis : string, default 'both'
            can be 'both' (default), 'x', or 'y' to
            control which set of gridlines are drawn.
        kwargs : any
            Other keyword arguments for grid.

        Examples
        --------
        .. plot:: pyplots/grid.py
            :include-source:

        """
        self._sp.grid(grid_on, axis=axis, **kwargs)

    def text(self, *args, **kwargs):
        """Create a Text instance at x, y with string text.

        Parameters
        ----------
        Any of the following keyword arguments is supported.

        Text properties:

            .. list-table::
              :header-rows: 1

              * - Property
                - Description
              * - alpha
                - float (0.0 transparent through 1.0 opaque)
              * - animated
                - [True | False]
              * - backgroundcolor
                - one of possible color
              * - bbox
                - rectangle prop dict
              * - color
                - one of possible color
              * - family or fontfamily or fontname or name
                - [FONTNAME | 'serif' | 'sans-serif' | 'cursive' | 'fantasy' | 'monospace' ]
              * - horizontalalignment or ha
                - [ 'center' | 'right' | 'left' ]
              * - label
                - string or anything printable with '%s' conversion.
              * - linespacing
                - float (multiple of font size)
              * -  position
                - (x,y)
              * -  rasterized
                - [True | False | None]
              * -  rotation
                - [ angle in degrees | 'vertical' | 'horizontal' ]
              * -  size or fontsize
                - [size in points | 'xx-small' | 'x-small' | 'small' | 'medium' |
                  'large' | 'x-large' | 'xx-large' ]
              * - stretch or fontstretch
                - [a numeric value in range 0-1000 | 'ultra-condensed' | 'extra-condensed'
                  | 'condensed' | 'semi-condensed' | 'normal' | 'semi-expanded' | 'expanded'
                  | 'extra-expanded' | 'ultra-expanded' ]
              * - style or fontstyle
                - [ 'normal' | 'italic' | 'oblique']
              * -  text
                - string or anything printable with '%s' conversion.
              * - verticalalignment or va or ma
                - [ 'center' | 'top' | 'bottom' | 'baseline' ]
              * -  visible
                - [True | False]
              * -  weight or fontweight
                - [a numeric value in range 0-1000 | 'ultralight' | 'light' | 'normal'
                  | 'regular' | 'book' | 'medium' | 'roman' | 'semibold' | 'demibold'
                  | 'demi' | 'bold' | 'heavy' | 'extra bold' | 'black' ]
              * - x
                - float, x position of the text.
              * - y
                - float. y position of the text.
              * - zorder
                - any number, objects with lower zorder values are drawn first.

        Font properties:

            .. list-table::
              :header-rows: 1

              * - Property
                - Description
              * - family
                - (font name or font family) es:  'serif', 'sans-serif', 'cursive',
                  'fantasy', or 'monospace'
              * - style
                - either between 'normal', 'italic' or 'oblique'
              * - variant
                - 'normal' or 'small-caps'
              * - stretch
                - A numeric value in the range 0-1000 or one of 'ultra-condensed',
                  'extra-condensed', 'condensed', 'semi-condensed', 'normal', 'semi-expanded',
                  'expanded', 'extra-expanded' or 'ultra-expanded'
              * - weight
                - A numeric value in the range 0-1000 or one of 'ultralight', 'light',
                  'normal', 'regular', 'book', 'medium', 'roman', 'semibold', 'demibold',
                  'demi', 'bold', 'heavy', 'extra bold', 'black'
              * - size
                - Either an relative value of 'xx-small', 'x-small', 'small', 'medium',
                  'large', 'x-large', 'xx-large' or an absolute font size, e.g., 12

        """
        if 'fontsize' not in kwargs:
            kwargs['fontsize'] = self._params['font.size']
        return self._sp.text(*args, **kwargs)

    def legend(self, *args, **kwargs):
        """Create legend for plot.

        Parameters
        ----------
        loc: integer or string or pair of floats, default: 0

            .. list-table::
              :header-rows: 1

              * - Integer
                - Location
              * - 0
                - 'best'
              * - 1
                - 'upper right'
              * - 2
                - 'upper left'
              * - 3
                - 'lower left'
              * - 4
                - 'lower right'
              * - 5
                - 'right'
              * - 6
                - 'center left'
              * - 7
                - 'center right'
              * - 8
                - 'lower center'
              * - 9
                - 'upper center'
              * - 10
                - 'center'

        bbox_to_anchor : tuple of floats
            Specify any arbitrary location for the legend in bbox_transform
            coordinates (default Axes coordinates). For example, to put the
            legend's upper right hand corner in the center of the axes the
            following keywords can be used: loc='upper right',
            bbox_to_anchor=(0.5, 0.5).
        ncol : integer
            The number of columns that the legend has. Default is 1.
        prop : None or dict
            The font properties of the legend. If None (default), the current
            default parameters will be used.
        fontsize : int or float or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
            Controls the font size of the legend. If the value is numeric
            the size will be the absolute font size in points.
            String values are relative to the current default font size.
            This argument is only used if prop is not specified.
        numpoints : None or int
            The number of marker points in the legend when creating a
            legend entry for a line. Default is None which will take the
            value from the legend.numpoints default parameter.
        scatterpoints : None or int
            The number of marker points in the legend when creating a
            legend entry for a scatter plot. Default is None which will
            take the value from the legend.scatterpoints default parameter.
        scatteryoffsets : iterable of floats
            The vertical offset (relative to the font size) for the markers
            created for a scatter plot legend entry. 0.0 is at the base the
            legend text, and 1.0 is at the top. To draw all markers at the
            same height, set to [0.5]. Default [0.375, 0.5, 0.3125].
        markerscale : None or int or float
            The relative size of legend markers compared with the originally
            drawn ones. Default is None which will take the value from the
            legend.markerscale default parameter.
        frameon : None or bool
            Control whether a frame should be drawn around the legend.
            Default is None which will take the value from the legend.frameon
            default parameter.
        fancybox : None or bool
            Control whether round edges should be enabled around the
            FancyBboxPatch which makes up the legend's background.
            Default is None which will take the value from the
            legend.fancybox default parameter.
        shadow : None or bool
            Control whether to draw a shadow behind the legend.
            Default is None which will take the value from the
            legend.shadow default parameter.
        framealpha : None or float
            Control the alpha transparency of the legend's frame.
            Default is None which will take the value from the
            legend.framealpha default parameter.
        mode : either between {"expand", None}
            If mode is set to "expand" the legend will be horizontally
            expanded to fill the axes area (or bbox_to_anchor if
            defines the legend's size).
        bbox_transform : None or matplotlib.transforms.Transform
            The transform for the bounding box (bbox_to_anchor).
            For a value of None (default) the Axes' transAxes transform
            will be used.
        title : str or None
            The legend's title. Default is no title (None).
        borderpad : float or None
            The fractional whitespace inside the legend border.
            Measured in font-size units. Default is None which will take
            the value from the legend.borderpad default parameter.
        labelspacing : float or None
            The vertical space between the legend entries. Measured in
            font-size units. Default is None which will take the value
            from the legend.labelspacing default parameter.
        handlelength : float or None
            The length of the legend handles. Measured in
            font-size units. Default is None which will take the
            value from the legend.handlelength default parameter.
        handletextpad : float or None
            The pad between the legend handle and text. Measured in
            font-size units. Default is None which will take the value
            from the legend.handletextpad default parameter.
        borderaxespad : float or None
            The pad between the axes and legend border. Measured in
            font-size units. Default is None which will take the value
            from the legend.borderaxespad default parameter.
        columnspacing : float or None
            The spacing between columns. Measured in font-size units.
            Default is None which will take the value from the
            legend.columnspacing default parameter.
        *args, **kwargs
            Same as :meth:`.text`.

        Examples
        --------
        .. plot:: pyplots/legend.py
            :include-source:

        """
        if 'fontsize' not in kwargs:
            kwargs['fontsize'] = self._params['font.size']
        self.show_legend = True
        return self._sp.legend(*args, **kwargs)

    def get_legend(self):
        """Returns the handler of current subplot legend."""
        return self._sp.get_legend()

    def title(self, text, *args, **kwargs):
        """Set a title for subplot."""
        if 'fontsize' not in kwargs:
            kwargs['fontsize'] = self._params['font.size']
        return self._sp.set_title(text, *args, **kwargs)

    def plot_path(self, path, path_style='-', path_width=1.5, path_color='k',
                  straight=False, start_style='h', start_facecolor='r',
                  start_edgecolor='k', start_edgewidth=1,
                  final_style='*', final_facecolor='g',
                  final_edgecolor='k', final_edgewidth=1):
        """Plot a path traversed by a point.

        By default, path is drawn in solid black, start point
        is drawn with a red star and the end point is drawn
        with a green asterisk.

        Parameters
        ----------
        path : CArray
            Every row contain one point coordinate.
        path_style : str
            Style for the path line. Default solid (-).
        path_width : int
            Width of path line. Default 1.5.
        path_color : str
            Color for the path line. Default black (k).
        straight : bool, default False
            If True, path will be plotted straight between start and end point.
        start_style : str
            Style for the start point. Default an hexagon (h).
        start_facecolor : str
            Color for the start point. Default red (r).
        start_edgecolor : str
            Color for the edge of the start point marker. Default black (k).
        start_edgewidth : scalar
            Width of the edge for the start point. Default 1.
        final_style : str
            Style for the end point. Default a star (*).
        final_facecolor : str
            Color for the end point. Default red (g).
        final_edgecolor : str
            Color for the edge of the final point marker. Default black (k).
        final_edgewidth : scalar
            Width of the edge for the end point. Default 1.

        Examples
        --------
        .. plot:: pyplots/plot_path.py
            :include-source:

        """
        path_2d = CArray(path).atleast_2d()
        if path_2d.shape[1] != 2:
            raise ValueError("cannot plot a {:}-Dimensional path."
                             "".format(path_2d.shape[1]))
        # Plotting full path, then the start and the end points
        if straight is False:
            self.plot(path_2d[:, 0], path_2d[:, 1],
                      linestyle=path_style,
                      color=path_color,
                      linewidth=path_width)
        else:
            self.plot(path_2d[[0, -1], 0], path_2d[[0, -1], 1],
                      linestyle=path_style, color=path_color)
        self.plot(path_2d[0, 0], path_2d[0, 1], marker=start_style,
                  markerfacecolor=start_facecolor,
                  markeredgecolor=start_edgecolor,
                  markeredgewidth=start_edgewidth)
        self.plot(path_2d[-1, 0], path_2d[-1, 1], marker=final_style,
                  markerfacecolor=final_facecolor,
                  markeredgecolor=final_edgecolor,
                  markeredgewidth=final_edgewidth)

    def imshow(self, img, *args, **kwargs):
        """Plot image.

        Parameters
        ----------
        img : CArray or PIL.Image.Image
            Image to plot.

        """
        if isinstance(img, CArray):
            img = img.tondarray()

        return self._sp.imshow(img, *args, **kwargs)

    def matshow(self, array, *args, **kwargs):
        """Plot an array as a matrix.

        Parameters
        ----------
        array : CArray
            Array that we want plot as a matrix.

        """
        return self._sp.matshow(array.tondarray(), *args, **kwargs)

    def quiver(self, U, V, X=None, Y=None,
               color='k', linestyle='-', linewidth=1.0, alpha=1.0):
        """A quiver plot displays velocity vectors as arrows
        with components (u,v) at the points (x,y).

        For example, the first vector is defined by components
        u(1), v(1) and is displayed at the point x(1), y(1).

        quiver(x,y,u,v) plots vectors as arrows at the coordinates
        specified in each corresponding pair of elements in x and y.

        quiver(u,v) draws vectors specified by u and v at equally
        spaced points in the x-y plane.

        Parameters
        ----------
        U, V: scalar or CArray
            Give the x and y components of the arrow vectors.
        X, Y: scalar or CArray, optional
            The x and y coordinates of the arrow locations
            (default is tail of arrow; see pivot kwarg)
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

        """
        if X is None:
            self._sp.quiver(U.tondarray(), V.tondarray(),
                            color=color, linestyle=linestyle,
                            linewidth=linewidth, alpha=alpha)
        else:
            self._sp.quiver(X.tondarray(), Y.tondarray(),
                            U.tondarray(), V.tondarray(),
                            color=color, linestyle=linestyle,
                            linewidth=linewidth, alpha=alpha)
