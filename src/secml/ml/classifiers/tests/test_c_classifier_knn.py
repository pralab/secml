from secml.ml.classifiers.tests import CClassifierTestCases

from six.moves import range
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from secml.ml.classifiers import CClassifierKNN
from secml.array import CArray
from secml.data.loader import CDLRandom
from secml.ml.peval.metrics import CMetricAccuracy


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

    def test_plot_dataset(self):
        self.logger.info("Draw the dataset... ")

        ds = CDLRandom(n_samples=100, n_classes=2, n_features=2,
                       n_redundant=0, n_informative=1, n_clusters_per_class=1,
                       random_state=10000).load()

        self.logger.info("Initializing KNeighbors Classifier... ")
        self.dataset = ds[:50, :]
        self.test = ds[50:, :]

        self.knn = CClassifierKNN(n_neighbors=3)
        self.knn.fit(self.dataset)

        x_min = self.dataset.X[:, 0].min() - 1
        x_max = self.dataset.X[:, 0].max() + 1
        y_min = self.dataset.X[:, 1].min() - 1
        y_max = self.dataset.X[:, 1].max() + 1
        self.step = 0.08
        xx, yy = np.meshgrid(np.arange(x_min, x_max, self.step),
                             np.arange(y_min, y_max, self.step))
        grid = CArray(np.c_[xx.ravel(), yy.ravel()])
        lab, Z_tree = self.knn.predict(grid, return_decision_function=True)
        Z_tree = Z_tree[:, 1]
        Z_tree = Z_tree.reshape(xx.shape)
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
        # cs = pl.contourf(xx, yy, Z_tree.data, 50)
        # cs = pl.contour(cs, levels=[0],colors = 'k', hold='on')
        plt.pcolormesh(xx, yy, Z_tree.get_data(), cmap=cmap_light)

        plt.scatter(self.dataset.X.get_data()[:, 0].ravel(),
                    self.dataset.X.get_data()[:, 1].ravel(),
                    c=self.dataset.Y.get_data(), marker='o', cmap=cmap_bold)
        plt.show()

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
        num_samp = 3
        with self.timer():
            dist, index_n, corresp = self.knn.kneighbors(
                single_sample, num_samp)
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
        self.logger.info(
            "Test for decision_function() and predict() methods.")

        scores_d = self._test_fun_multiclass(self.knn, self.dataset.todense())
        scores_s = self._test_fun_multiclass(self.knn, self.dataset.tosparse())

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
