from secml.ml.classifiers.tests import CClassifierTestCases

from secml.ml.classifiers import CClassifierKNN
from secml.data.loader import CDLRandom, CDLRandomBlobs
from secml.ml.peval.metrics import CMetricAccuracy
from secml.utils import fm


class TestCClassifierKNN(CClassifierTestCases):
    """Unit test for CClassifierKNN."""

    def setUp(self):

        ds = CDLRandom(n_samples=100, n_classes=3, n_features=2,
                       n_redundant=0, n_informative=2, n_clusters_per_class=1,
                       random_state=10000).load()

        self.dataset = ds[:50, :]
        self.test = ds[50:, :]

        self.logger.info("Initializing KNeighbors Classifier... ")
        self.knn = CClassifierKNN(n_neighbors=3)
        self.knn.fit(self.dataset)

    def test_plot(self):
        ds = CDLRandomBlobs(n_samples=100, centers=3, n_features=2,
                            random_state=1).load()
        fig = self._test_plot(self.knn, ds, levels=[0.5])
        fig.savefig(fm.join(fm.abspath(__file__), 'figs',
                            'test_c_classifier_knn.pdf'))

    def test_classification(self):
        self.logger.info("Check the classification method... ")

        lab_cl, score = self.knn.predict(
            self.test.X, return_decision_function=True)

        acc = CMetricAccuracy().performance_score(self.test.Y, lab_cl)

        self.logger.info("Real label:\n{:}".format(self.test.Y.tolist()))
        self.logger.info("Predicted label:\n{:}".format(lab_cl.tolist()))

        self.logger.info("Accuracy: {:}".format(acc))

        self.assertGreater(acc, 0.9)

    def test_kneighbors(self):
        single_sample = self.test.X[0, :]
        array_samples = self.test.X[1:11, :]

        self.logger.info("Checking KNN classifier on a single sample...")
        with self.timer():
            dist, index_n, corresp = self.knn.kneighbors(single_sample)
        self.logger.info("Sample to evaluate: {:}".format(single_sample))
        self.logger.info("")
        self.logger.info("Closest: {:}, index {:}, distance {:}"
                         "".format(corresp[dist.argmin(), :],
                                   index_n[dist.argmin()],
                                   dist.min()))

        self.logger.info("Checking KNN classifier on multiple samples...")
        num_samp = 2
        with self.timer():
            dist, index_n, corresp = self.knn.kneighbors(
                array_samples, num_samp)
        for i in range(10):
            self.logger.info("Sample to evaluate: {:}".format(single_sample))
            self.logger.info("Closest: {:}, index {:}, distance {:}"
                             "".format(corresp[i, :], index_n[i], dist[i, :]))

    def test_fun(self):
        """Test for decision_function() and predict() methods."""
        scores_d = self._test_fun(self.knn, self.dataset.todense())
        scores_s = self._test_fun(self.knn, self.dataset.tosparse())

        self.assert_array_almost_equal(scores_d, scores_s)

    def test_preprocess(self):
        """Test classifier with preprocessors inside."""
        knn = CClassifierKNN(n_neighbors=3)
        # All linear transformations
        self._test_preprocess(self.dataset, knn,
                              ['min-max', 'mean-std'],
                              [{'feature_range': (-1, 1)}, {}])

        # Mixed linear/nonlinear transformations
        self._test_preprocess(self.dataset, knn,
                              ['pca', 'unit-norm'], [{}, {}])


if __name__ == '__main__':
    CClassifierTestCases.main()
