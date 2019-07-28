"""
.. module:: CPlotConstraint
   :synopsis: Plot constraint bounds on bi-dimensional feature spaces

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>

"""
from secml.figure._plots import CPlotFunction
from secml.optim.constraints import CConstraint


class CPlotConstraint(CPlotFunction):
    """Plot constraint on bi-dimensional feature spaces.

    See Also
    --------
    .CPlot : basic subplot functions.
    .CFigure : creates and handle figures.

    """

    def plot_constraint(self, constraint, grid_limits=None, n_grid_points=30):
        """Plot constraint bound.

        Parameters
        ----------
        constraint : CConstraint
            Constraint to be plotted.
        grid_limits : list of tuple
            List with a tuple of min/max limits for each axis.
            If None, [(0, 1), (0, 1)] limits will be used.
        n_grid_points : int, optional
            Number of grid points. Default 30.

        """
        if not isinstance(constraint, CConstraint):
            raise TypeError(
                "'constraint' must be an instance of `CConstraint`.")

        self.plot_fun(func=constraint.constraint,
                      plot_background=False,
                      grid_limits=grid_limits,
                      n_grid_points=n_grid_points,
                      levels=[0],
                      levels_linewidth=1.5)
