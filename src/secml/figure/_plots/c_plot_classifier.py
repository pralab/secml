"""
.. module:: CPlotClassifier
   :synopsis: Plot a classifier's decision regions on 2D feature spaces.

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml.figure._plots import CPlotFunction
from secml.ml.classifiers import CClassifier
from secml.array import CArray


class CPlotClassifier(CPlotFunction):
    """Plot a classifier.

    Custom plotting parameters can be specified.
    Currently parameters default:
     - grid: False.

    See Also
    --------
    .CPlot : basic subplot functions.
    .CFigure : creates and handle figures.

    """

    def apply_params_clf(self):
        """Apply defined parameters to active subplot."""
        self.grid(grid_on=False)

    def plot_decision_regions(self, clf, plot_background=True,
                              grid_limits=None, n_grid_points=30, cmap='jet'):
        """Plot decision boundaries and regions for the given classifier.

        Parameters
        ----------
        clf : CClassifier
            Classifier which decision function should be plotted.
        plot_background : bool, optional
            Specifies whether to color the decision regions. Default True.
            in the background using a colorbar.
        grid_limits : list of tuple
            List with a tuple of min/max limits for each axis.
            If None, [(0, 1), (0, 1)] limits will be used.
        n_grid_points : int, optional
            Number of grid points. Default 30.
        cmap : str or list or `matplotlib.pyplot.cm`
            Colormap to use (default 'jet'). Could be a list of colors.

        """
        if not isinstance(clf, CClassifier):
            raise TypeError("'clf' must be an instance of `CClassifier`.")

        self.plot_fun(func=clf.predict,
                      multipoint=True,
                      colorbar=False,
                      n_colors=clf.n_classes,
                      cmap=cmap,
                      levels=CArray.arange(0.5, clf.n_classes).tolist(),
                      plot_background=plot_background,
                      grid_limits=grid_limits,
                      n_grid_points=n_grid_points,
                      alpha=0.5)

        self.apply_params_clf()
