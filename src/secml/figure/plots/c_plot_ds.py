from secml.figure.plots import CPlot
from secml.core.type_utils import is_list
from matplotlib import cm


class CPlotDataset(CPlot):
    """Plots a Dataset.

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
    .CDataset : store and manage a dataset.
    .CPlot : basic subplot functions.
    .CFigure : creates and handle figures.

    """
    class_type = 'ds'

    def __init__(self, sp, default_params=None):

        # Calling CPlot constructor
        super(CPlotDataset, self).__init__(
            sp=sp, default_params=default_params)

        # Specific plot parameters (use `set_params` to alter)
        self.show_legend = True
        self.grid(grid_on=True)

    def _apply_params(self):
        """Apply defined parameters to active subplot."""
        fig_legend = self.get_legend()
        if fig_legend is not None:
            fig_legend.set_visible(self.show_legend)

    def plot_ds(self, dataset, colors=None, markers='o', *args, **kwargs):
        """Plot patterns of each class with a different color/marker.

        Parameters
        ----------
        dataset : CDataset
            Dataset that contain samples which we want plot.
        colors : list or None, optional
            Color to be used for plotting each class.
            If not None, the number of input colors should be
            always equal to the number of dataset's classes.
            If None and the number of classes is 1, blue will be used.
            If None and the number of classes is 2, blue and red will be used.
            If None and the number of classes is > 2, 'jet' colormap is used.
        markers : list or str, optional
            Marker to use for plotting. Default is 'o' (circle).
            If a string, the same specified marker will be used for each class.
            If a list, must specify one marker for each dataset's class.
        args, kwargs : any
            Any optional argument for plots.
            If the number of classes is 2, a `plot` will be created.
            If the number of classes is > 2, a `scatter` plot will be created.

        """
        classes = dataset.classes
        if colors is None:
            if classes.size <= 2:
                colors = ['b', 'r']
            else:  # Next returns an ndarray classes.size X 4 (RGB + Alpha)
                colors = cm.ScalarMappable(
                    cmap='jet').to_rgba(xrange(classes.size))
        else:
            if len(colors) != classes.size:
                raise ValueError(
                    "{:} markers must be specified.".format(classes.size))

        if is_list(markers) and len(markers) != classes.size:
            raise ValueError(
                "{:} markers must be specified.".format(classes.size))

        for cls_idx, cls in enumerate(classes.tolist()):
            c = colors[cls_idx]
            m = markers[cls_idx] if is_list(markers) else markers
            this_c_p = dataset.Y.find(dataset.Y == cls)
            self.plot(dataset.X[this_c_p, 0], dataset.X[this_c_p, 1],
                      linestyle='None', color=c, marker=m, *args, **kwargs)

        # Customizing figure
        self._apply_params()
