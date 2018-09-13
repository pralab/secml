"""
Created on 27/apr/2015

@author: davidemaiorca

This module provides the test for the CConstraintL1
"""
import unittest
from prlib.utils import CUnitTest

from prlib.array import CArray
from prlib.figure import CFigure
from prlib.data.loader import CDataLoader
from prlib.optimization.constraints import CConstraint
from prlib.optimization.constraints import CConstraintL1


class TestCConstraintL1(CUnitTest):
    """Unit test for CConstraintL1."""

    def setUp(self):
        """Test setup."""
        self.logger.info("Generating training set... ")
        loader = CDataLoader.create('random_blobs')
        self.dataset = loader.load()
        self.center = 0
        self.radius = 1
        # TODO: create class in our lib to wrap artificial data generators
        self.constraint = CConstraintL1(CArray([-0.3, 1.7]), 3)

    def test_instantiation(self):
        """Test constraint instantiation."""
        self.logger.info("Testing CConstraintL1 instantiation method...")
        self.assertEqual(CConstraint.create(
            'l1', self.center, self.radius).__class__.__name__,
                         self.constraint.__class__.__name__)

    def test_l1_plot(self):
        """Tests the graphical appearance of the constraint"""
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

    def test_binary_projection(self):
        n_dim = 100
        radius = 20
        loader = CDataLoader.create('random_binary',
                                    n_samples=100,
                                    n_features=n_dim)
        dataset = loader.load()
        center = dataset.X[0, :]
        point = dataset.X[50, :]
        cons = CConstraintL1(center=center, radius=radius)
        with self.timer():
            new_point = cons.projection(point)
        self.logger.info(
            "Initial Distance: " + str(cons.constraint(point) + radius))
        self.logger.info("Center: " + str(center))
        self.logger.info("Point: " + str(point))
        self.logger.info("Projection: " + str(new_point))
        self.logger.info("Final Distance: " + str(
            cons.constraint(new_point) + radius) + "  radius: " + str(radius))


if __name__ == '__main__':
    unittest.main()
