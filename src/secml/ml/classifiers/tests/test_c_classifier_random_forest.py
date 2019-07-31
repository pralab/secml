from secml.ml.classifiers.tests import CClassifierTestCases

from secml.data.loader import CDLIris, CDLRandomBlobs
from secml.ml.classifiers import CClassifierRandomForest
from secml.utils import fm


class TestCClassifierRandomForest(CClassifierTestCases):
    """Unit test for CRandomForest."""

    def setUp(self):
        self.dataset = CDLIris().load()

        self.rnd_forest = CClassifierRandomForest(random_state=0)

    def test_classify(self):
        self.logger.info("Testing random forest training ")
        self.rnd_forest.fit(self.dataset)

        self.logger.info("Testing classification with trees")

        self.logger.info(
            "Number of classes: {:}".format(self.rnd_forest.n_classes))

        y, result = self.rnd_forest.predict(
            self.dataset.X[0, :], return_decision_function=True)
        self.logger.info(
            "Probability of affinity to each class: {:}".format(result))
        self.logger.info("Class of affinity: {:}".format(y))
        self.assertEqual(self.dataset.Y[0], y, "Wrong classification")

        y, result = self.rnd_forest.predict(
            self.dataset.X[50, :], return_decision_function=True)
        self.logger.info(
            "Probability of affinity to each class: {:}".format(result))
        self.logger.info("Class of affinity: {:}".format(y))
        self.assertEqual(self.dataset.Y[50], y, "Wrong classification")

        y, result = self.rnd_forest.predict(
            self.dataset.X[120, :], return_decision_function=True)
        self.logger.info(
            "Probability of affinity to each class: {:}".format(result))
        self.logger.info("Class of affinity: {:}".format(y))
        self.assertEqual(self.dataset.Y[120], y, "Wrong classification")

    def test_fun(self):
        """Test for decision_function() and predict() methods."""
        scores_d = self._test_fun(self.rnd_forest, self.dataset.todense())
        scores_s = self._test_fun(self.rnd_forest, self.dataset.tosparse())

        self.assert_array_almost_equal(scores_d, scores_s)

    def test_preprocess(self):
        """Test classifier with preprocessors inside."""
        # All linear transformations
        self._test_preprocess(self.dataset, self.rnd_forest,
                              ['min-max', 'mean-std'],
                              [{'feature_range': (-1, 1)}, {}])

        # Mixed linear/nonlinear transformations
        self._test_preprocess(self.dataset, self.rnd_forest,
                              ['pca', 'unit-norm'], [{}, {}])

    def test_plot(self):
        """ Compare the classifiers graphically"""
        ds = CDLRandomBlobs(n_samples=100, centers=3, n_features=2,
                            random_state=1).load()
        fig = self._test_plot(self.rnd_forest, ds, levels=[0.5])
        fig.savefig(fm.join(fm.abspath(__file__), 'figs',
                            'test_c_classifier_random_forest.pdf'))


if __name__ == '__main__':
    CClassifierTestCases.main()
