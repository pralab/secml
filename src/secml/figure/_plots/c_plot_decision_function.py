"""
.. module:: CPlotDecisionFunction
   :synopsis: Plot classifier's decision regions on 2D feature spaces.

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>

"""
from secml.figure._plots import CPlotFunction
from secml.ml.classifiers import CClassifier
from secml.array import CArray


class CPlotDecisionFunction(CPlotFunction):
    """Plot a classifier's decision regions on bi-dimensional feature spaces.

    Custom plotting parameters can be specified.
    Currently parameters default:
     - grid: False.

    See Also
    --------
    .CPlot : basic subplot functions.
    .CFigure : creates and handle figures.

    """

    def apply_params(self):
        """Apply defined parameters to active subplot."""
        self.grid(grid_on=False)

    def plot_decision_function(self, clf, n_grid_points=100,
                              plot_background=True, cmap='jet'):
        """Plot decision boundaries and regions for the given classifier."""

        if not isinstance(clf, CClassifier):
            raise TypeError(
                "plot_constraint requires CClassifier as input!")

        self.plot_fobj(func=clf.predict,
                       multipoint=True,
                       colorbar=False,
                       n_colors=clf.n_classes,
                       cmap=cmap,
                       levels=CArray.arange(0.5, clf.n_classes, 1).tolist(),
                       plot_background=plot_background,
                       n_grid_points=n_grid_points,
                       alpha=0.5)

        # Customizing figure
        self.apply_params()
