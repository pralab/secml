import unittest
from secml.utils import CUnitTest

from secml.optimization.constraints import CConstraintBox, CConstraintL1, CConstraintL2, CConstraintSet
from secml.array import CArray
import numpy as np


class TestCOptimizer(CUnitTest):
    """Test for COptimizer class."""

    def test_minimize(self):
        """Testing function minimization."""
        self.logger.info(
            "Test for Class of the management of the constraints ... ")
        self.cons_set = CConstraintSet()
        consl1 = CConstraintL1(center=[0, 0], radius=2)
        consbox = CConstraintBox(ub=0.5, lb=0)
        self.cons_set.insert_constraint(consl1)
        self.cons_set.insert_constraint(consbox)

        point = CArray([1, 1], dtype=np.float)
        self.logger.info("ConsL1 violated : " + str(consl1.is_violated(point)))
        self.logger.info(
            "ConsBox violated : " + str(consbox.is_violated(point)))

        self.logger.info(
            "ConSet violated : " + str(self.cons_set.is_violated(point)))

        self.logger.info(
            "ConSet projection : " + str(self.cons_set.projection(point)))


if __name__ == "__main__":
    unittest.main()
