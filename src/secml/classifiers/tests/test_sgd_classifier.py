"""
Created on 3/may/2015
Class to test CStochasticGradientDescentClassifier

@author: Paolo Russu
If you find any BUG, please notify authors first.
"""
import unittest
from prlib.utils import CUnitTest
from prlib.classifiers import CClassifierSGD, CClassifierSVM
from prlib.classifiers.regularizer import *
from prlib.classifiers.loss import *
from prlib.array import CArray
from prlib.data.loader import CDLRandom, CDLRandomBlobs
from prlib.features.normalization import CNormalizerMinMax
from prlib.peval.metrics import CMetric
from prlib.figure.c_figure import CFigure


class TestSGDClassifier(CUnitTest):
    """Unit test for SGD Classifier."""

    def setUp(self):
        """Test for init and train methods."""        
        # generate synthetic data
        self.dataset = CDLRandom(n_features=1000, n_redundant=200,
                                 n_informative=250,
                                 n_clusters_per_class=2).load()

        self.dataset.X = CNormalizerMinMax().train_normalize(self.dataset.X)

        self.logger.info("Testing classifier creation ")
        self.sgd = CClassifierSGD(regularizer=CRegularizerL2(),
                                  loss=CLossHinge(),
                                  n_iter=5000)

        self.logger.info("Testing SGD classifier training ")

        self.sgd.train(self.dataset)

    def test_time(self):
        """ Compare execution time of SGD and SVM"""
        self.logger.info("Testing training speed of SGD compared to SVM ")
        self.svm = CClassifierSVM()

        with self.timer() as t_svm:
            self.svm.train(self.dataset)
        self.logger.info("Execution time of SVM: " + str(t_svm.interval) + "\n")
        with self.timer() as t_sgd:
            self.sgd.train(self.dataset)
        self.logger.info("Execution time of SGD: " + str(t_sgd.interval) + "\n")

    def test_draw(self):
        """ Compare the classifiers graphically"""
        self.logger.info("Testing classifiers graphically")

        # generate 2D synthetic data
        dataset = CDLRandom(n_features=2, n_redundant=1, n_informative=1,
                            n_clusters_per_class=1).load()
        dataset.X = CNormalizerMinMax().train_normalize(dataset.X)

        self.sgd.train(dataset)

        self.svm = CClassifierSVM()
        self.svm.train(dataset)

        fig = CFigure(width=10, markersize=8)
        fig.subplot(2, 1, 1, sp_type='ds')
        # Plot dataset points
        fig.sp.plot_ds(dataset)
        # Plot objective function
        fig.switch_sptype(sp_type='function')
        fig.sp.plot_fobj(self.svm.discriminant_function,
                         grid_limits=dataset.get_bounds())
        fig.sp.title('SVM')

        fig.subplot(2, 1, 2, sp_type='ds')
        # Plot dataset points
        fig.sp.plot_ds(dataset)
        # Plot objective function
        fig.switch_sptype(sp_type='function')
        fig.sp.plot_fobj(self.sgd.discriminant_function,
                         grid_limits=dataset.get_bounds())
        fig.sp.title('SGD Classifier')

        fig.show()

    def test_performance(self):
        """ Compare the classifiers performance"""
        self.logger.info("Testing error performance of the "
                         "classifiers on the training set")

        self.svm = CClassifierSVM()
        self.svm.train(self.dataset)

        label_svm, y_svm = self.svm.classify(self.dataset.X)
        label_sgd, y_sgd = self.sgd.classify(self.dataset.X)

        acc_svm = CMetric.create('f1').performance_score(
            self.dataset.Y, label_svm)
        acc_sgd = CMetric.create('f1').performance_score(
            self.dataset.Y, label_sgd)

        self.logger.info("Accuracy of SVM: {:}".format(acc_svm))
        self.assertGreater(acc_svm, 0.90, "Accuracy of SVM: {:}".format(acc_svm))
        self.logger.info("Accuracy of SGD: {:}".format(acc_sgd))
        self.assertGreater(acc_sgd, 0.90, "Accuracy of SGD: {:}".format(acc_sgd))

    def test_margin(self):

        self.logger.info("Testing margin separation of PRASGD...")

        import numpy as np

        # we create 50 separable points
        dataset = CDLRandomBlobs(n_samples=50, centers=2, random_state=0,
                                 cluster_std=0.60).load()

        # fit the model
        clf = CClassifierSGD(loss=CLossHinge(), n_iter=200,
                             regularizer=CRegularizerL2(), alpha=0.01)
        clf.train(dataset)

        # plot the line, the points, and the nearest vectors to the plane
        xx = CArray.linspace(-1, 5, 10)
        yy = CArray.linspace(-1, 5, 10)

        X1, X2 = np.meshgrid(xx.tondarray(), yy.tondarray())
        Z = CArray.empty(X1.shape)
        for (i, j), val in np.ndenumerate(X1):
            x1 = val
            x2 = X2[i, j]
            Z[i, j] = clf.discriminant_function([[x1, x2]])
        levels = [-1.0, 0.0, 1.0]
        linestyles = ['dashed', 'solid', 'dashed']
        colors = 'k'
        fig = CFigure(linewidth=1)
        fig.sp.contour(X1, X2, Z, levels, colors=colors, linestyles=linestyles)
        fig.sp.scatter(dataset.X[:, 0], dataset.X[:, 1], c=dataset.Y, s=40)

        fig.show()


if __name__ == '__main__':
    unittest.main()
