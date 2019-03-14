"""
Created on 27/apr/2015

@author: davidemaiorca
"""
import unittest
from secml.utils import CUnitTest

from secml.data import CDataset
from sklearn import datasets
from secml.optim.constraints import CConstraint, CConstraintL2


class TestCConstraintBox(CUnitTest):
    """Unit test for CKernelLinear."""

    def setUp(self):
        # TODO: create class in our lib to wrap artificial data generators
        patterns, labels = datasets.make_classification(n_features=2,
                                                        n_redundant=0,
                                                        n_informative=1,
                                                        n_clusters_per_class=1)

        self.dataset = CDataset(patterns, labels)
        self.constraint = CConstraintL2()
           
    def test_instantiation(self):
        """Test constraint instantiation."""
        self.logger.info("Testing CConstraintBox instantiation method")
        self.assertEqual(CConstraint.create('l2').__class__.__name__,
                         self.constraint.__class__.__name__)

    def test_projection(self):
        """Test projection onto feasible box."""
        # FIXME: This test should also verify something...
        self.logger.info("Testing projection on box")
        self.logger.info(self.dataset.X[0, :])
        self.logger.info(self.constraint.constraint(self.dataset.X))
        self.logger.info(self.constraint.projection(self.dataset.X[0, :]))
        self.logger.info(self.constraint.constraint(self.dataset.X[0, :]))

if __name__ == '__main__':
    unittest.main()
