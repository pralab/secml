from secml.ml.classifiers.tests import CClassifierTestCases

from secml.data.loader import CDLRandom
from secml.ml.classifiers import CClassifierKDE
from secml.array import CArray
from secml.ml.kernel import CKernelRBF
from secml.ml.features.normalization import CNormalizerMinMax
from secml.figure import CFigure


class TestCClassifierKDE(CClassifierTestCases):
    """Unit test for CClassifierKDE."""

    def setUp(self):
        """Test for init and fit methods."""
        # generate synthetic data
        self.dataset = CDLRandom(n_features=2, n_redundant=0, n_informative=1,
                                 n_clusters_per_class=1, random_state=0).load()

        self.dataset.X = CNormalizerMinMax().fit_transform(self.dataset.X)

        self.logger.info("Testing classifier creation ")
        k = CKernelRBF(gamma=1e-3)
        # k = CKernelLinear
        self.kde = CClassifierKDE(k)

        self.logger.info(
            "Testing Stochastic gradient descent classifier training ")

    def test_plot(self):
        """ Compare the classifiers graphically"""
        fig = self._test_plot(self.kde, self.dataset)
        fig.savefig('test_c_classifier_kde.pdf')

    def test_fun(self):
        """Test for decision_function() and predict() methods."""
        self._test_fun(self.kde, self.dataset.todense())
        self._test_fun(self.kde, self.dataset.tosparse())

    def test_gradient(self):
        """Unittest for `gradient_f_x` method."""
        i = 5  # IDX of the point to test
        pattern = self.dataset.X[i, :]
        self.logger.info("P {:}: {:}".format(i, pattern))

        self.logger.info("Testing dense data...")
        ds = self.dataset.todense()
        self.kde.fit(ds)

        # Run the comparison with numerical gradient
        # (all classes will be tested)
        grads_d = self._test_gradient_numerical(self.kde, pattern.todense())

        self.logger.info("Testing sparse data...")
        ds = self.dataset.tosparse()
        self.kde.fit(ds)

        # Run the comparison with numerical gradient
        # (all classes will be tested)
        grads_s = self._test_gradient_numerical(self.kde, pattern.tosparse())

        # Compare dense gradients with sparse gradients
        for grad_i, grad in enumerate(grads_d):
            self.assert_array_almost_equal(
                grad.atleast_2d(), grads_s[grad_i])

    def test_preprocess(self):
        """Test classifier with preprocessors inside."""
        ds = CDLRandom().load()

        # All linear transformations with gradient implemented
        self._test_preprocess(ds, self.kde,
                              ['min-max', 'mean-std'],
                              [{'feature_range': (-1, 1)}, {}])
        self._test_preprocess_grad(ds, self.kde,
                                   ['min-max', 'mean-std'],
                                   [{'feature_range': (-1, 1)}, {}])

        self.logger.info("The following case will skip the gradient test")
        # Mixed linear/nonlinear transformations without gradient
        self._test_preprocess(ds, self.kde, ['pca', 'unit-norm'], [{}, {}])


if __name__ == '__main__':
    CClassifierTestCases.main()
