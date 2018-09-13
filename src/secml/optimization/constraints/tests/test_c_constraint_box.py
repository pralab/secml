"""
Created on 27/apr/2015
This class implements tests for the CConstraintBox
@author: Davide Maiorca

"""
import unittest
from prlib.utils import CUnitTest

from prlib.data import CDataset
from prlib.data.loader import CDataLoader
from prlib.array import CArray
from prlib.optimization.constraints import CConstraint
from prlib.optimization.constraints import CConstraintBox
from prlib.figure import CFigure


class TestCConstraintBox(CUnitTest):
    """Unit test for CConstraintBox."""

    def setUp(self):
        """Sets up the test."""
        self.logger.info("Generating training set... ")
        loader = CDataLoader.create('random_blobs')
        self.dataset = loader.load()
        # self.lb = -1
        self.lb = [-1, -0.5]
        # self.ub = 2
        self.ub = [1, 2]
        self.constraint = CConstraintBox(lb=self.lb, ub=self.ub)

    def test_instantiation(self):
        """Test constraint instantiation."""
        self.logger.info("Testing CConstraintBox instantiation method")
        self.assertEqual(CConstraint.create('box', self.lb, self.ub).__class__.__name__,
                         self.constraint.__class__.__name__)

#     #FIXME: Implement meaningful tests
#     def test_projection(self):
#         """Test projection onto feasible box."""
#         self.logger.info("Testing projection on box")
#         print self.dataset.X[0]
#         print self.constraint.constraint(self.dataset.X)
#         print self.constraint.projection(self.dataset.X[0])
#         print self.constraint.constraint(self.dataset.X[0])

    def test_box_plot(self):
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


if __name__ == '__main__':
    unittest.main()
