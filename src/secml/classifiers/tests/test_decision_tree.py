"""
Created on 27/apr/2015
Class to test CDecisionTree

@author: Davide Maiorca
If you find any BUG, please notify authors first.
"""
import unittest
from prlib.utils import CUnitTest

from prlib.data.loader import CDLRandomToy
from prlib.classifiers import CClassifierDecisionTree


class TestCDecisionTree(CUnitTest):
    """Unit test for CDecisionTree."""

    def setUp(self):
        self.dataset = CDLRandomToy('iris').load()

        self.dec_tree = CClassifierDecisionTree()

    def test_classify(self):
        """Test for classify method. """
        self.logger.info("Testing decision tree classifier training ")
        self.dec_tree.train(self.dataset)

        self.logger.info("Testing classification with trees")

        self.logger.info(
            "Number of classes: {:}".format(self.dec_tree.n_classes))

        y, result = self.dec_tree.classify(self.dataset.X[0, :])
        self.logger.info(
            "Probability of affinity to each class: {:}".format(result))
        self.logger.info("Class of affinity: {:}".format(y))
        self.assertEquals(y, self.dataset.Y[0], "Wrong classification")

        y, result = self.dec_tree.classify(self.dataset.X[50, :])
        self.logger.info(
            "Probability of affinity to each class: {:}".format(result))
        self.logger.info("Class of affinity: {:}".format(y))
        self.assertEquals(y, self.dataset.Y[50], "Wrong classification")

        y, result = self.dec_tree.classify(self.dataset.X[120, :])
        self.logger.info(
            "Probability of affinity to each class: {:}".format(result))
        self.logger.info("Class of affinity: {:}".format(y))
        self.assertEquals(y, self.dataset.Y[120], "Wrong classification")


if __name__ == '__main__':
    unittest.main()
