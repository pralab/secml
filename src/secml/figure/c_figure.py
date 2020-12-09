"""
.. module:: CFigure
   :synopsis: A figure for making plots.

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from secml.core import CCreator
from secml.figure._plots import CPlot
from secml.utils import LastInDict
from secml.core.type_utils import is_tuple

import os
import matplotlib as mpl
if os.name == 'posix' and os.environ.get('DISPLAY', '') == '':
    # If no display is available, use file-only backend
    mpl.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class CFigure(CCreator):
    """Creates a Figure.

    A Figure is a collection of subplots. The last active subplot can be
    accessed by calling CFigure.sp`, followed by the desired plotting
    function (plot, scatter, contour, etc.).
    Each subplot is identified by an index (grid slot) inside an imaginary
    grid (n_rows, n_columns, grid_slot), counting from left to right,
    from top to bottom. By default, a subplot is created in a single-row,
    single-column imaginary grid (1, 1, 1).

    Parameters
    ----------
    height : scalar, optional
        Height of the new figure. Default 6.
    width : scalar, optional
        Width of the new figure. Default 6.
    title : str, optional
        Super title of the new figure. This is not the subplot title.
        To set a title for active subplot use :meth:`.subtitle`.
        Default is to not set a super title.
    linewidth : float, optional
        Define default linewidth for all subplots. Default 2.
    fontsize : int, optional
        Define default fontsize for all subplots. Default 12.
    markersize : scalar, optional
        Define default markersize for all subplots. Default 7.

    Notes
    -----
    Not all `matplotlib` methods and/or parameters are currently available.
    If needed, directly access the `matplotlib.Axes` active subplot instance
    through the `CFigure.sp._sp` parameter.

    Examples
    --------
    >>> from secml.figure import CFigure

    >>> fig = CFigure(fontsize=14)
    >>> fig.sp.plot([0, 1], color='red')

    >>> fig.show()  # This will open a new window with the figure

    """
    def __init__(self, height=6, width=6, title="",
                 fontsize=12, linewidth=2, markersize=7):

        # Instancing figure with desired dimensions
        self.width = width
        self.height = height
        self._fig = plt.figure(figsize=(self.width, self.height))

        # Setting default fontsize, linewidth, markersize
        self._default_params = {'font.size': fontsize,
                                'lines.linewidth': linewidth,
                                'lines.markersize': markersize}

        # Setting figure super title
        self.title(title)

        # Setting up the subplots container
        self._sp_data = LastInDict()

        # Handle of the subplot grid
        self._gs = None

    @property
    def sp(self):
        """Return reference to active subplot class.

        If no subplot is available, creates a new standard subplot
        in `(1,1,1)` position and returns its reference.

        """
        if self.n_sp == 0:
            self.subplot(1, 1, 1)
        return self._sp_data.lastin

    @property
    def n_sp(self):
        """Returns the number of created subplots."""
        return len(self._sp_data)

    def get_state(self):
        raise NotImplementedError

    def set_state(self, state_dict, copy=False):
        raise NotImplementedError

    def load_state(self, path):
        raise NotImplementedError

    def save_state(self, path):
        raise NotImplementedError

    def subplot(self, n_rows=1, n_cols=1, grid_slot=1, **kwargs):
        """Create a new subplot into specific position.

        Creates a new subplot placing it in the n_plot position of the
        n_rows * n_cols imaginary grid. Position is indexed in raster-scan.

        If subplot is created in a slot occupied by another subplot,
         old subplot will be used.

        Parameters
        ----------
        n_rows : int
            Number of rows of the imaginary grid used for
            subdividing the figure. Default 1 (one row).
        n_cols : int
            Number of columns of the imaginary grid used for
            subdividing the figure. Default 1 (one column).
        grid_slot : int or tuple
            If int, raster scan position of the new subplot. Default 1.
            If tuple, index of the slot of the grid. Each dimension
            can be specified as an int or a slice,
            e.g. in a 3 x 3 subplot grid, grid_slot=(0, slice(1, 3))
            will create a subplot at row index 0 that spans between
            columns index 1 and 2.

        Examples
        --------
        .. plot:: pyplots/subplot.py
            :include-source:

        """
        # Create a new grid if shape has changed or this is the first grid
        if self._gs is None or self._gs.get_geometry()[0] != n_rows or \
                self._gs.get_geometry()[1] != n_cols:
            self._gs = gridspec.GridSpec(n_rows, n_cols)
        # If grid_slot is not a tuple, assume we want to use a single slot
        grid_slot = grid_slot-1 if not is_tuple(grid_slot) else grid_slot

        # Calling matplotlib subplot switcher
        axes = self._fig.add_subplot(self._gs[grid_slot], **kwargs)

        # Set default parameters
        axes.tick_params(labelsize=self._default_params['font.size'])

        sp_id = hex(id(axes))  # Index of the subplot

        # Create the subplot if not available or switch lastitem reference
        if sp_id not in self._sp_data:
            self._sp_data[sp_id] = CPlot(
                sp=axes, default_params=self._default_params)
        else:
            self._sp_data.lastin_key = sp_id

        return self.sp

    def get_default_params(self):
        """Return current defaults for the figure.

        Returns
        -------
        default_parameters : dict
            Contain default parameters value set.

        """
        return self._default_params

    @staticmethod
    def show(block=True):
        """Show all created figures.

        Parameters
        ----------
        block : boolean, default True
            If true, execution is halted until the showed
            figure(s) are closed.

        """
        plt.show(block=block)

    def close(self, fig=None):
        """Close current or input figure.

        Parameters
        ----------
        fig : CFigure or None
            Handle to figure to close. If None (default),
            current figure is closed.

        """
        plt.close(self._fig if fig is None else fig)

    def subplots_adjust(self, left=0.125, right=0.9,
                        bottom=0.1, top=0.9, wspace=0.2, hspace=0.2):
        """Tune the subplot layout.

        Parameters
        ----------
        left : float, default 0.125
            Left side of the subplots.
        right : float, default 0.9
            Right side of the subplots.
        bottom : float, default 0.1
            Bottom of the subplots.
        top : float, default  0.9
            Top of the subplots.
        wspace : float, default 0.2
            Amount of width reserved for blank space between subplots.
        hspace : float, default 0.2
            Amount of height reserved for white space between subplots

        Examples
        --------
        .. plot:: pyplots/subplots_adjust.py
            :include-source:

        """
        self._fig.subplots_adjust(left=left, bottom=bottom,
                                  right=right, top=top,
                                  wspace=wspace, hspace=hspace)

    def tight_layout(self, pad=1.08, h_pad=None, w_pad=None, rect=None):
        """Adjust space between plot and figure.

        Parameters
        ----------
        pad : float, default 1.08
            Padding between the figure edge and the edges of
            subplots, as a fraction of the font-size.
        h_pad, w_pad : float, defaults to pad_inches.
            padding (height/width) between edges of adjacent subplots.
        rect : tuple of scalars, default is (0, 0, 1, 1).
            (left, bottom, right, top) in the normalized figure coordinate
            that the whole subplots area (including labels) will fit into.

        """
        self._fig.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad, rect=rect)

    def title(self, label, **kwargs):
        """Set the global title for current figure.

        Parameters
        ----------
        label : str
            Text to use for the title.
        **kwargs
            Same as :meth:`.text` method.

        """
        if 'fontsize' not in kwargs:
            kwargs['fontsize'] = self.get_default_params()['font.size']
        return self._fig.suptitle(label, **kwargs)

    def savefig(self, fname, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', file_format=None, transparent=False,
                bbox_inches=None, bbox_extra_artists=None, pad_inches=0.1):
        """Save figure to disk.
        
        Parameters
        ----------      
        fname : string 
            containing a path to a filename, or a Python file-like object.
            If file_format is None and fname is a string, the output
            file_format is deduced from the extension of the filename.
            If the filename has no extension, the value of the rc parameter
            savefig.file_format is used.
            If fname is not a string, remember to specify file_format
            to ensure that the correct backend is used.
        dpi : [ None | scalar > 0 ], optional
            The resolution in dots per inch. If None it will default to the
            value savefig.dpi in the matplotlibrc file.
        facecolor, edgecolor : color or str, optional
            The colors of the figure rectangle. Default 'w' (white).
        orientation: [ 'landscape' | 'portrait' ], optional
            not supported on all backends; currently only on postscript output
        file_format : str, optional
            One of the file extensions supported by the active backend. Most
            backends support png, pdf, ps, eps and svg.
        transparent : bool, optional
            If True, the axes patches will all be transparent;
            the figure patch will also be transparent unless facecolor
            and/or edgecolor are specified via kwargs. This is useful,
            for example, for displaying a plot on top of a colored
            background on a web page.
            The transparency of these patches will be restored to their
            original values upon exit of this function.
        bbox_inches : scalar or str, optional
            Bbox in inches. Only the given portion of the figure is saved.
            If 'tight', try to figure out the tight bbox of the figure.
        bbox_extra_artists : list
            A list of extra artists that will be considered when the tight
            bbox is calculated.
        pad_inches : scalar
            Amount of padding around the figure when bbox_inches is 'tight'.

        """
        self._fig.savefig(fname, dpi=dpi, facecolor=facecolor,
                          edgecolor=edgecolor, orientation=orientation,
                          format=file_format, transparent=transparent,
                          bbox_inches=bbox_inches, pad_inches=pad_inches,
                          bbox_extra_artists=bbox_extra_artists)
