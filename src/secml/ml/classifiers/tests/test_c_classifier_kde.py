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

    def test_draw(self):
        """ Compare the classifiers graphically"""
        self.logger.info("Testing classifiers graphically")
        # Preparation of the grid
        fig = CFigure()
        fig.sp.plot_ds(self.dataset)

        self.kde.fit(self.dataset)

        fig.sp.plot_fobj(self.kde.decision_function, y=1, plot_levels=False)
        fig.title('kde Classifier')

        self.logger.info(self.kde.predict(self.dataset.X))

        fig.savefig('test_c_classifier_kde.pdf')

    def test_fun(self):
        """Test for decision_function() and predict() methods."""
        self._test_fun(self.kde, self.dataset.todense())
        self._test_fun(self.kde, self.dataset.tosparse())

    def test_gradient(self):
        """Unittest for `gradient_f_x` method."""
        self.kde.fit(self.dataset)

        import random
        pattern = CArray(random.choice(self.dataset.X.get_data()))
        self.logger.info("Randomly selected pattern:\n%s", str(pattern))

        # Comparison with numerical gradient
        self._test_gradient_numerical(self.kde, pattern)

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
