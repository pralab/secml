"""
Created on 19/oct/2015
Class to test CClassifierKernelDensityEstimator

@author: Ambra Demontis
If you find any BUG, please notify authors first.
"""
import unittest
from secml.utils import CUnitTest

from secml.data.loader import CDLRandom
from secml.classifiers import CClassifierKDE
from secml.figure import CFigure
from secml.kernel import CKernelRBF
from secml.features.normalization import CNormalizerMinMax


class TestCClassifierKernelDensityEstimator(CUnitTest):
    """Unit test for CClassifierKDE."""

    def setUp(self):
        """Test for init and train methods."""
        # generate synthetic data
        self.dataset = CDLRandom(n_features=2, n_redundant=0, n_informative=1,
                                 n_clusters_per_class=1).load()

        self.dataset.X = CNormalizerMinMax().train_normalize(self.dataset.X)

        self.logger.info("Testing classifier creation ")
        k = CKernelRBF(gamma=1e1)
        # k = CKernelLinear
        self.kde = CClassifierKDE(k)

        self.logger.info("Testing Stochastic gradient descent classifier training ")

    def test_draw(self):
        """ Compare the classifiers graphically"""
        self.logger.info("Testing classifiers graphically")
        # Preparation of the grid
        fig = CFigure()
        fig.switch_sptype(sp_type='ds')
        fig.sp.plot_ds(self.dataset)

        self.kde.train(self.dataset)

        fig.switch_sptype(sp_type='function')
        fig.sp.plot_fobj(self.kde.discriminant_function, label=1)
        fig.title('kde Classifier')

        self.logger.info(self.kde.classify(self.dataset.X))

        fig.show()


if __name__ == '__main__':
    unittest.main()
