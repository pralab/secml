"""
Created on 27/apr/2015
This class implements tests for the CConstraintBox

@author: Davide Maiorca
@author: Battista Biggio

"""
import unittest
from secml.utils import CUnitTest

from secml.data import CDataset
from secml.data.loader import CDataLoader
from secml.array import CArray
from secml.optim.constraints import CConstraint
from secml.optim.constraints import CConstraintBox
from secml.figure import CFigure
from secml.utils import fm


class TestCConstraintBox(CUnitTest):
    """Unit test for CConstraintBox."""

    def setUp(self):
        """Sets up the test."""
        self.logger.info("Generating training set... ")
        self.filename = 'test_c_constraint_box.pdf'
        loader = CDataLoader.create('blobs')
        self.dataset = loader.load()
        self.lb = [-1, -0.5]
        self.ub = [1, 2]
        self.constraint = CConstraintBox(lb=self.lb, ub=self.ub)

    def test_instantiation(self):
        """Test constraint instantiation."""
        self.logger.info("Testing CConstraintBox instantiation method")
        self.assertEqual(
            CConstraint.create('box', self.lb, self.ub).__class__.__name__,
            self.constraint.__class__.__name__)

    def test_projection(self):
        """Test projection onto feasible box."""
        self.logger.info("Testing projection on box")
        x = CArray(self.lb) - 1.0
        self.assertTrue(self.constraint.is_violated(x))
        x1 = self.constraint.projection(x)
        self.assertFalse(self.constraint.is_violated(x1))

    def test_sparse(self):
        """Test using sparse arrays."""
        x = CArray(self.lb) - 1.0
        x = x.tosparse()
        self.assertTrue(self.constraint.is_violated(x))

        # try when lb and ub are scalars, and x is all zeros.
        box = CConstraintBox(lb=-1, ub=1)
        x = CArray([0, 0])
        x = x.tosparse()
        self.assertFalse(box.is_violated(x))

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
        fig.savefig(fm.join(fm.abspath(__file__), self.filename),
                    file_format='pdf')


if __name__ == '__main__':
    unittest.main()
