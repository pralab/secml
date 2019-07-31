from secml.ml.classifiers.tests import CClassifierTestCases

from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier

from secml.array import CArray
from secml.figure import CFigure
from secml.data.loader import CDLRandom
from secml.ml.classifiers import CClassifierMCSLinear, CClassifierSVM
from secml.ml.peval.metrics import CMetric


class TestCClassifierMCSLinear(CClassifierTestCases):
    """Unit test for CClassifierMCSLinear."""

    def setUp(self):
        self.dataset = CDLRandom(n_samples=1000, n_features=500,
                                 n_redundant=0, n_informative=50,
                                 n_clusters_per_class=1,
                                 random_state=0).load()

        self.mcs = CClassifierMCSLinear(CClassifierSVM(),
                                        num_classifiers=10,
                                        max_features=0.5,
                                        max_samples=0.5,
                                        random_state=0)

    def test_classification(self):
        with self.timer():
            self.mcs.fit(self.dataset)
            self.logger.info("Trained MCS.")

        with self.timer():
            self.sklearn_bagging = BaggingClassifier(SVC(kernel='linear'),
                                                     n_estimators=10,
                                                     max_samples=0.5,
                                                     max_features=0.5,
                                                     bootstrap=False,
                                                     random_state=0)
            self.sklearn_bagging.fit(self.dataset.X.get_data(),
                                     self.dataset.Y.tondarray())
            self.logger.info("Trained Sklearn Bagging + SVC.")

        label_mcs, s_mcs = self.mcs.predict(
            self.dataset.X, return_decision_function=True)
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
        ds = CDLRandom(n_features=2, n_redundant=0, n_informative=1,
                       n_clusters_per_class=1).load()
        fig = self._test_plot(self.mcs, ds)
        fig.savefig('test_c_classifier_mcs_linear.pdf')

    def test_fun(self):
        """Test for decision_function() and predict() methods."""
        self._test_fun(self.mcs, self.dataset.todense())
        self._test_fun(self.mcs, self.dataset.tosparse())

    def test_gradient(self):
        """Unittest for `gradient_f_x` method."""
        i = 5  # IDX of the point to test
        pattern = self.dataset.X[i, :]
        self.logger.info("P {:}: {:}".format(i, pattern))

        self.logger.info("Testing dense data...")
        ds = self.dataset.todense()
        self.mcs.fit(ds)

        # Run the comparison with numerical gradient
        # (all classes will be tested)
        grads_d = self._test_gradient_numerical(self.mcs, pattern.todense())

        self.logger.info("Testing sparse data...")
        ds = self.dataset.tosparse()
        self.mcs.fit(ds)

        # Run the comparison with numerical gradient
        # (all classes will be tested)
        grads_s = self._test_gradient_numerical(self.mcs, pattern.tosparse())

        # Compare dense gradients with sparse gradients
        for grad_i, grad in enumerate(grads_d):
            self.assert_array_almost_equal(
                grad.atleast_2d(), grads_s[grad_i])

    def test_preprocess(self):
        """Test classifier with preprocessors inside."""
        ds = CDLRandom().load()

        # All linear transformations with gradient implemented
        self._test_preprocess(ds, self.mcs,
                              ['min-max', 'mean-std'],
                              [{'feature_range': (-1, 1)}, {}])
        self._test_preprocess_grad(ds, self.mcs,
                                   ['min-max', 'mean-std'],
                                   [{'feature_range': (-1, 1)}, {}])

        # Mixed linear/nonlinear transformations without gradient
        self._test_preprocess(ds, self.mcs, ['pca', 'unit-norm'], [{}, {}])


if __name__ == '__main__':
    CClassifierTestCases.main()
