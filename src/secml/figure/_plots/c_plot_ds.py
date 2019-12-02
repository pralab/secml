"""
.. module:: CPlotDataset
   :synopsis: Dataset plots.

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from matplotlib import cm

from secml.figure._plots import CPlot
from secml.core.type_utils import is_list


class CPlotDataset(CPlot):
    """Plots a Dataset.

    Custom plotting parameters can be specified.

    Currently parameters default:
     - show_legend: True
     - grid: True

    See Also
    --------
    .CDataset : store and manage a dataset.
    .CPlot : basic subplot functions.
    .CFigure : creates and handle figures.

    """

    def apply_params_ds(self):
        """Apply defined parameters to active subplot."""
        fig_legend = self.get_legend()
        if self.show_legend is not False and fig_legend is not None:
            fig_legend.set_visible(True)
        self.grid(grid_on=True)

    def plot_ds(self, dataset, colors=None, markers='o', *args, **kwargs):
        """Plot patterns of each class with a different color/marker.

        Parameters
        ----------
        dataset : CDataset
            Dataset that contain samples which we want plot.
        colors : list or None, optional
            Color to be used for plotting each class.
            If a list, each color will be assigned to a dataset's class,
            with repetitions if necessary.
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
            if classes.size <= 6:
                colors = ['blue', 'red', 'lightgreen', 'black', 'gray', 'cyan']
                from matplotlib.colors import ListedColormap
                cmap = ListedColormap(colors[:classes.size])
            else:
                cmap = 'jet'
        else:
            from matplotlib.colors import ListedColormap
            cmap = ListedColormap(colors)

        # Next returns an ndarray classes.size X 4 (RGB + Alpha)
        colors = cm.ScalarMappable(
            cmap=cmap).to_rgba(range(classes.size))

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
        self.apply_params_ds()
