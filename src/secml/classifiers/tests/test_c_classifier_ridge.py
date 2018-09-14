import unittest
from secml.utils import CUnitTest
from secml.classifiers import CClassifierRidge, CClassifierSVM
from secml.data.loader import CDLRandom
from secml.features.normalization import CNormalizerMinMax
from secml.peval.metrics import CMetric
from secml.figure.c_figure import CFigure


class TestRidgeClassifier(CUnitTest):
    """Unit test for Ridge Classifier."""

    def setUp(self):
        """Test for init and train methods."""
        # generate synthetic data
        self.dataset = CDLRandom(n_features=1000, n_redundant=200,
                                 n_informative=250,
                                 n_clusters_per_class=2).load()

        self.dataset.X = CNormalizerMinMax().train_normalize(self.dataset.X)

        self.logger.info("Testing classifier creation ")
        self.ridge = CClassifierRidge(alpha=1e-6)

        self.logger.info("Testing ridge classifier training ")
        self.ridge.train(self.dataset)

    def test_time(self):
        """ Compare execution time of ridge and SVM"""
        self.logger.info("Testing training speed of ridge compared to SVM ")
        self.svm = CClassifierSVM()

        with self.timer() as t_svm:
            self.svm.train(self.dataset)
        self.logger.info(
            "Execution time of SVM: {:}".format(t_svm.interval))
        with self.timer() as t_ridge:
            self.ridge.train(self.dataset)
        self.logger.info(
            "Execution time of ridge: {:}".format(t_ridge.interval))

    def test_draw(self):
        """ Compare the classifiers graphically"""
        self.logger.info("Testing classifiers graphically")

        # generate 2D synthetic data
        dataset = CDLRandom(n_features=2, n_redundant=0, n_informative=2,
                            n_clusters_per_class=1).load()
        dataset.X = CNormalizerMinMax().train_normalize(dataset.X)

        self.ridge.train(dataset)

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
        fig.sp.plot_fobj(self.ridge.discriminant_function,
                         grid_limits=dataset.get_bounds())
        fig.sp.title('ridge Classifier')

        fig.show()

    def test_performance(self):
        """ Compare the classifiers performance"""
        self.logger.info("Testing error performance of the "
                         "classifiers on the training set")

        self.svm = CClassifierSVM()
        self.svm.train(self.dataset)

        label_svm, y_svm = self.svm.classify(self.dataset.X)
        label_ridge, y_ridge = self.ridge.classify(self.dataset.X)

        acc_svm = CMetric.create('f1').performance_score(
            self.dataset.Y, label_svm)
        acc_ridge = CMetric.create('f1').performance_score(
            self.dataset.Y, label_ridge)

        self.logger.info("Accuracy of SVM: {:}".format(acc_svm))
        self.assertGreater(acc_svm, 0.90,
                           "Accuracy of SVM: {:}".format(acc_svm))
        self.logger.info("Accuracy of ridge: {:}".format(acc_ridge))
        self.assertGreater(acc_ridge, 0.90,
                           "Accuracy of ridge: {:}".format(acc_ridge))


if __name__ == '__main__':
    unittest.main()
