from secml.ml.classifiers.tests import CClassifierTestCases

from secml.data.loader import CDLRandom
from secml.ml.classifiers import CClassifierLogistic
from secml.ml.features.normalization import CNormalizerMinMax
from secml.utils import fm


class TestCClassifierLogistic(CClassifierTestCases):
    """Unit test for CClassifierLogistic."""

    def setUp(self):
        """Test for init and fit methods."""
        # generate synthetic data
        self.dataset = CDLRandom(n_features=2, n_redundant=0, n_informative=1,
                                 n_clusters_per_class=1, random_state=99).load()

        self.dataset.X = CNormalizerMinMax().fit_transform(self.dataset.X)

        self.logger.info("Testing classifier creation ")
        
        self.log = CClassifierLogistic(random_state=99)

    def test_plot(self):
        """ Compare the classifiers graphically"""
        fig = self._test_plot(self.log, self.dataset)
        fig.savefig(fm.join(fm.abspath(__file__), 'figs',
                            'test_c_classifier_logistic.pdf'))

    def test_fun(self):
        """Test for decision_function() and predict() methods."""
        scores_d = self._test_fun(self.log, self.dataset.todense())
        scores_s = self._test_fun(self.log, self.dataset.tosparse())

        self.assert_array_almost_equal(scores_d, scores_s)

    def test_gradient(self):
        """Unittests for gradient_f_x."""
        self.logger.info("Testing log.gradient_f_x() method")

        i = 5  # IDX of the point to test
        pattern = self.dataset.X[i, :]
        self.logger.info("P {:}: {:}".format(i, pattern))

        self.logger.info("Testing dense data...")
        ds = self.dataset.todense()
        self.log.fit(ds)

        # Run the comparison with numerical gradient
        # (all classes will be tested)
        grads_d = self._test_gradient_numerical(self.log, pattern.todense())

        self.logger.info("Testing sparse data...")
        ds = self.dataset.tosparse()
        self.log.fit(ds)

        # Run the comparison with numerical gradient
        # (all classes will be tested)
        grads_s = self._test_gradient_numerical(self.log, pattern.tosparse())

        # Compare dense gradients with sparse gradients
        for grad_i, grad in enumerate(grads_d):
            self.assert_array_almost_equal(
                grad.atleast_2d(), grads_s[grad_i])

    def test_sparse(self):
        """Test classifier operations on sparse data."""

        self._test_sparse_linear(self.dataset.tosparse(), self.log)

    def test_preprocess(self):
        """Test classifier with preprocessors inside."""
        ds = CDLRandom().load()

        # All linear transformations with gradient implemented
        self._test_preprocess(ds, self.log,
                              ['min-max', 'mean-std'],
                              [{'feature_range': (-1, 1)}, {}])
        self._test_preprocess_grad(ds, self.log,
                                   ['min-max', 'mean-std'],
                                   [{'feature_range': (-1, 1)}, {}])

        self.logger.info("The following case will skip the gradient test")
        # Mixed linear/nonlinear transformations without gradient
        self._test_preprocess(ds, self.log, ['pca', 'unit-norm'], [{}, {}])


if __name__ == '__main__':
    CClassifierTestCases.main()
