"""
.. module:: CPlotConstraint
   :synopsis: Plot constraint bounds on bi-dimensional feature spaces

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>

"""
from secml.figure._plots import CPlotFunction
from secml.optim.constraints import CConstraint


class CPlotConstraint(CPlotFunction):
    """Plot constraint bounds on bi-dimensional feature spaces.

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

    def plot_constraint(self, constraint, n_grid_points=100):
        """Plot constraint bound."""

        if not isinstance(constraint, CConstraint):
            raise TypeError("plot_constraint requires CConstraint as input!")

        self.plot_fobj(func=constraint.constraint,
                       plot_background=False,
                       n_grid_points=n_grid_points,
                       levels=[0],
                       levels_linewidth=1.5)

        # Customizing figure
        self.apply_params()
