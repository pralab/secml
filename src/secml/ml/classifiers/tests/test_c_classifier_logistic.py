from secml.ml.classifiers.tests import CClassifierTestCases

from secml.data.loader import CDLRandom
from secml.ml.classifiers import CClassifierLogistic
from secml.array import CArray
from secml.ml.features.normalization import CNormalizerMinMax
from secml.figure import CFigure


class TestCClassifierLogistic(CClassifierTestCases):
    """Unit test for CClassifierLogistic."""

    def setUp(self):
        """Test for init and fit methods."""
        # generate synthetic data
        self.dataset = CDLRandom(n_features=2, n_redundant=0, n_informative=1,
                                 n_clusters_per_class=1, random_state=99).load()

        self.dataset.X = CNormalizerMinMax().fit_transform(self.dataset.X)

        self.logger.info("Testing classifier creation ")
        
        self.log = CClassifierLogistic(random_seed=99)

    def test_draw(self):
        """ Compare the classifiers graphically"""
        self.logger.info("Testing classifiers graphically")
        # Preparation of the grid
        fig = CFigure()
        fig.sp.plot_ds(self.dataset)

        self.log.fit(self.dataset)

        fig.sp.plot_fobj(self.log.decision_function, y=1)
        fig.title('Logistic Classifier')

        self.logger.info(self.log.predict(self.dataset.X))

        fig.show()

    def test_fun(self):
        """Test for decision_function() and predict() methods."""
        self._test_fun(self.log, self.dataset)

    def test_gradient(self):
        """Unittests for gradient_f_x."""
        self.logger.info("Testing log.gradient_f_x() method")

        i = 5  # IDX of the point to test

        # Randomly extract a pattern to test
        pattern = self.dataset.X[i, :]
        self.logger.info("P {:}: {:}".format(i, pattern))

        self.log.fit(self.dataset)

        # Run the comparison with numerical gradient
        # (all classes will be tested)
        self._test_gradient_numerical(self.log, pattern)

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
