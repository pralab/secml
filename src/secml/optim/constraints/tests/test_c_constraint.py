from secml.utils import CUnitTest
from abc import ABCMeta, abstractmethod

from secml.figure import CFigure

from numpy import *


class CConstraintTestCases(object):
    """Wrapper for TestCConstraint to make unittest.main() work correctly."""

    class TestCConstraint(CUnitTest):
        """Unit test for CConstraint."""
        __metaclass__ = ABCMeta

        @abstractmethod
        def param_setter(self):
            pass

        def _plot_constraint(self):

            grid_limits = [(-3, 3), (-3, 3)]
            fig = CFigure(height=6, width=6)
            fig.switch_sptype(sp_type='function')
            fig.sp.plot_fobj(func=self.constraint.constraint,
                             grid_limits=grid_limits)
            fig.sp.plot_fobj(func=self.constraint.constraint,
                             plot_background=False,
                             n_grid_points=50,
                             grid_limits=grid_limits,
                             levels=[0])
            fig.show()

        def setUp(self):

            self._constr_creation()

            self._plot_constraint()

        # todo: add a plot that shows the constraint grafically

        # todo: add a check to understand if the a point near the constraint
        #  is correclty recognized as point inside/outside (if it is exaclty
        #  on the constraint for us is inside?)

        # todo: add a test that checks if after the projection the point is
        #  inside the constraint (both with the apposite function and
        #  checking the norm of the distance)