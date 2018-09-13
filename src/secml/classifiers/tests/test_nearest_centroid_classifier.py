"""
Created on 19/oct/2015
Class to test CClassifierKernelDensityEstimator

@author: Ambra Demontis
If you find any BUG, please notify authors first.
"""
import unittest
from prlib.utils import CUnitTest

from prlib.data.loader import CDLRandom
from prlib.classifiers import CClassifierNearestCentroid
from prlib.figure.c_figure import CFigure
from prlib.features.normalization import CNormalizerMinMax


class TestCClassifierNearestCentroid(CUnitTest):
    """Unit test for CClassifierNearestCentroid."""

    def setUp(self):
        """Test for init and train methods."""

        self.dataset = CDLRandom(n_features=2, n_redundant=0, n_informative=1,
                                 n_clusters_per_class=1).load()

        self.dataset.X = CNormalizerMinMax().train_normalize(self.dataset.X)

        self.nc = CClassifierNearestCentroid()

    def test_draw(self):
        """ Compare the classifiers graphically"""
        self.logger.info("Testing classifiers graphically")
        # Preparation of the grid
        fig = CFigure()
        fig.switch_sptype(sp_type='ds')
        fig.sp.plot_ds(self.dataset)

        self.nc.train(self.dataset)

        fig.switch_sptype(sp_type='function')
        fig.sp.plot_fobj(self.nc.discriminant_function, label=1)
        fig.title('nearest centroid  Classifier')

        self.logger.info(self.nc.classify(self.dataset.X))

        fig.show()


if __name__ == '__main__':
    unittest.main()
