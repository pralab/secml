import unittest
from secml.utils import CUnitTest

from secml.data.loader import CDLRandomToy
from secml.classifiers import CClassifierRandomForest


class TestCRandomForest(CUnitTest):
    """Unit test for CRandomForest."""

    def setUp(self):
        self.dataset = CDLRandomToy('iris').load()

        self.rnd_forest = CClassifierRandomForest()
       
    def test_classify(self):

        self.logger.info("Testing random forest training ")
        self.rnd_forest.train(self.dataset)

        self.logger.info("Testing classification with trees")
        
        self.logger.info(
            "Number of classes: {:}".format(self.rnd_forest.n_classes))
        
        y, result = self.rnd_forest.classify(self.dataset.X[0, :])
        self.logger.info(
            "Probability of affinity to each class: {:}".format(result))
        self.logger.info("Class of affinity: {:}".format(y))
        self.assertEquals(y, self.dataset.Y[0], "Wrong classification")
        
        y, result = self.rnd_forest.classify(self.dataset.X[50, :])
        self.logger.info(
            "Probability of affinity to each class: {:}".format(result))
        self.logger.info("Class of affinity: {:}".format(y))
        self.assertEquals(y, self.dataset.Y[50], "Wrong classification")
        
        y, result = self.rnd_forest.classify(self.dataset.X[120, :])
        self.logger.info(
            "Probability of affinity to each class: {:}".format(result))
        self.logger.info("Class of affinity: {:}".format(y))
        self.assertEquals(y, self.dataset.Y[120], "Wrong classification")


if __name__ == '__main__':
    unittest.main()
