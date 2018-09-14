import unittest
from secml.utils import CUnitTest
from secml.array import CArray
from secml.figure import CFigure
from secml.data.loader import CDLRandom
from secml.classifiers import CClassifierMCSLinear, CClassifierSVM
from secml.peval.metrics import CMetric

from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier


class TestCClassifierMCSLinear(CUnitTest):
    """Unit test for CClassifierMCSLinear."""

    def setUp(self):
        self.dataset = CDLRandom(n_samples=1000, n_features=500,
                                 n_redundant=0, n_informative=50,
                                 n_clusters_per_class=1).load()

    def test_classification(self):

        with self.timer():
            self.mcs = CClassifierMCSLinear(CClassifierSVM(),
                                            num_classifiers=10,
                                            max_features=0.5,
                                            max_samples=0.5)
            self.mcs.train(self.dataset)
            self.logger.info("Trained MCS.")

        with self.timer():
            self.sklearn_bagging = BaggingClassifier(SVC(),
                                                     n_estimators=10,
                                                     max_samples=0.5,
                                                     max_features=0.5,
                                                     bootstrap=False)
            self.sklearn_bagging.fit(self.dataset.X.get_data(),
                                     self.dataset.Y.tondarray())
            self.logger.info("Trained Sklearn Bagging + SVC.")

        label_mcs, s_mcs = self.mcs.classify(self.dataset.X)
        label_skbag = self.sklearn_bagging.predict(self.dataset.X.get_data())

        f1_mcs = CMetric.create('f1').performance_score(
            self.dataset.Y, label_mcs)
        f1_skbag = CMetric.create('f1').performance_score(
            self.dataset.Y, CArray(label_skbag))

        self.logger.info("F1-Score of MCS: {:}".format(f1_mcs))
        self.logger.info(
            "F1-Score of Sklearn Bagging + SVC: {:}".format(f1_skbag))
        self.assertLess(
            abs(f1_mcs - f1_skbag), 0.1,
            "Performance difference is: {:}".format(abs(f1_mcs - f1_skbag)))

    def test_plot(self):

        self.dataset = CDLRandom(n_features=2, n_redundant=0, n_informative=1,
                                 n_clusters_per_class=1).load()

        self.logger.info("Training MCS on 2D Dataset... ")
        self.mcs = CClassifierMCSLinear(CClassifierSVM(),
                                        max_features=0.5, max_samples=0.5)
        self.mcs.train(self.dataset)

        fig = CFigure()
        # Plot dataset points
        fig.switch_sptype(sp_type='ds')
        fig.sp.plot_ds(self.dataset)
        # Plot objective function
        fig.switch_sptype(sp_type='function')
        fig.sp.plot_fobj(self.mcs.discriminant_function,
                         grid_limits=self.dataset.get_bounds())
        fig.show()


if __name__ == '__main__':
    unittest.main()
