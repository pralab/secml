from secml.utils import CUnitTest

import numpy as np
import pylab as pl
from matplotlib.colors import ListedColormap

from secml.ml.classifiers import CClassifierKNN
from secml.data import CDataset
from secml.array import CArray
from secml.data.loader import CDLRandom
from secml.ml.peval.metrics import CMetricAccuracy


class TestCClassifierKNN(CUnitTest):
    """Unit test for CClassifierKNN."""

    def setUp(self):

        ds = CDLRandom(n_samples=100, n_classes=3, n_features=2,
                       n_redundant=0, n_informative=2, n_clusters_per_class=1,
                       random_state=10000).load()

        self.dataset = ds[:50, :]
        self.test = ds[50:, :]

        self.logger.info("Initializing KNeighbors Classifier... ")
        self.knn = CClassifierKNN(n_neighbors=3)
        self.knn.train(self.dataset)

    def test_plot_dataset(self):
        self.logger.info("Draw the dataset... ")

        ds = CDLRandom(n_samples=100, n_classes=2, n_features=2,
                       n_redundant=0, n_informative=1, n_clusters_per_class=1,
                       random_state=10000).load()

        self.logger.info("Initializing KNeighbors Classifier... ")
        self.dataset = ds[:50, :]
        self.test = ds[50:, :]

        self.knn = CClassifierKNN(n_neighbors=3)
        self.knn.train(self.dataset)

        x_min = self.dataset.X[:, 0].min() - 1
        x_max = self.dataset.X[:, 0].max() + 1
        y_min = self.dataset.X[:, 1].min() - 1
        y_max = self.dataset.X[:, 1].max() + 1
        self.step = 0.08
        xx, yy = np.meshgrid(np.arange(x_min, x_max, self.step),
                             np.arange(y_min, y_max, self.step))
        grid = CArray(np.c_[xx.ravel(), yy.ravel()])
        lab, Z_tree = self.knn.classify(grid)
        Z_tree = Z_tree[:, 1]
        Z_tree = Z_tree.reshape(xx.shape)
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
        # cs = pl.contourf(xx, yy, Z_tree.data, 50)
        # cs = pl.contour(cs, levels=[0],colors = 'k', hold='on')
        pl.pcolormesh(xx, yy, Z_tree.get_data(), cmap=cmap_light)

        pl.scatter(self.dataset.X.get_data()[:, 0].ravel(),
                   self.dataset.X.get_data()[:, 1].ravel(),
                   c=self.dataset.Y.get_data(), marker='o', cmap=cmap_bold)
        pl.show()

    def test_classification(self):
        self.logger.info("Check the classification method... ")

        lab_cl, score = self.knn.classify(self.test.X)

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
        for i in xrange(10):
            self.logger.info("Sample to evaluate: {:}".format(single_sample))
            self.logger.info("Closest: {:}, index {:}, distance {:}"
                             "".format(corresp[i, :], index_n[i], dist[i, :]))

    def test_fun(self):
        """Test for discriminant_function() and classify() methods."""
        self.logger.info(
            "Test for discriminant_function() and classify() methods.")

        def _check_df_scores(s, n_samples):
            self.assertEqual(type(s), CArray)
            self.assertTrue(s.isdense)
            self.assertEqual(1, s.ndim)
            self.assertEqual((n_samples,), s.shape)
            self.assertEqual(float, s.dtype)

        def _check_classify_scores(l, s, n_samples, n_classes):
            self.assertEqual(type(l), CArray)
            self.assertEqual(type(s), CArray)
            self.assertTrue(l.isdense)
            self.assertTrue(s.isdense)
            self.assertEqual(1, l.ndim)
            self.assertEqual(2, s.ndim)
            self.assertEqual((n_samples,), l.shape)
            self.assertEqual((n_samples, n_classes), s.shape)
            self.assertEqual(int, l.dtype)
            self.assertEqual(float, s.dtype)

        self.knn.train(self.dataset)

        x = x_norm = self.dataset.X
        p = p_norm = self.dataset.X[0, :].ravel()

        # Normalizing data if a normalizer is defined
        if self.knn.normalizer is not None:
            x_norm = self.knn.normalizer.normalize(x)
            p_norm = self.knn.normalizer.normalize(p)

        # Testing discriminant_function on multiple points

        df_scores_0 = self.knn.discriminant_function(x, y=0)
        self.logger.info(
            "discriminant_function(x, y=0):\n{:}".format(df_scores_0))
        _check_df_scores(df_scores_0, self.dataset.num_samples)

        df_scores_1 = self.knn.discriminant_function(x, y=1)
        self.logger.info(
            "discriminant_function(x, y=1):\n{:}".format(df_scores_1))
        _check_df_scores(df_scores_1, self.dataset.num_samples)

        df_scores_2 = self.knn.discriminant_function(x, y=2)
        self.logger.info(
            "discriminant_function(x, y=2):\n{:}".format(df_scores_2))
        _check_df_scores(df_scores_2, self.dataset.num_samples)

        # Testing _discriminant_function on multiple points

        ds_priv_scores_0 = self.knn._discriminant_function(x_norm, y=0)
        self.logger.info("_discriminant_function(x_norm, y=0):\n"
                         "{:}".format(ds_priv_scores_0))
        _check_df_scores(ds_priv_scores_0, self.dataset.num_samples)

        ds_priv_scores_1 = self.knn._discriminant_function(x_norm, y=1)
        self.logger.info("_discriminant_function(x_norm, y=1):\n"
                         "{:}".format(ds_priv_scores_1))
        _check_df_scores(ds_priv_scores_1, self.dataset.num_samples)

        ds_priv_scores_2 = self.knn._discriminant_function(x_norm, y=2)
        self.logger.info("_discriminant_function(x_norm, y=2):\n"
                         "{:}".format(ds_priv_scores_2))
        _check_df_scores(ds_priv_scores_2, self.dataset.num_samples)

        # Comparing output of public and private

        self.assertFalse((df_scores_0 != ds_priv_scores_0).any())
        self.assertFalse((df_scores_1 != ds_priv_scores_1).any())
        self.assertFalse((df_scores_2 != ds_priv_scores_2).any())

        # Testing classify on multiple points

        labels, scores = self.knn.classify(x)
        self.logger.info(
            "classify(x):\nlabels: {:}\nscores: {:}".format(labels, scores))
        _check_classify_scores(
            labels, scores, self.dataset.num_samples, self.knn.n_classes)

        # Comparing output of discriminant_function and classify

        self.assertFalse((df_scores_0 != scores[:, 0].ravel()).any())
        self.assertFalse((df_scores_1 != scores[:, 1].ravel()).any())
        self.assertFalse((df_scores_2 != scores[:, 2].ravel()).any())

        # Testing discriminant_function on single point

        df_scores_0 = self.knn.discriminant_function(p, y=0)
        self.logger.info("discriminant_function(p, y=0):\n"
                         "{:}".format(df_scores_0))
        _check_df_scores(df_scores_0, 1)

        df_scores_1 = self.knn.discriminant_function(p, y=1)
        self.logger.info("discriminant_function(p, y=1):\n"
                         "{:}".format(df_scores_1))
        _check_df_scores(df_scores_1, 1)

        df_scores_2 = self.knn.discriminant_function(p, y=2)
        self.logger.info("discriminant_function(p, y=2):\n"
                         "{:}".format(df_scores_2))
        _check_df_scores(df_scores_2, 1)

        # Testing _discriminant_function on single point

        df_priv_scores_0 = self.knn._discriminant_function(p_norm, y=0)
        self.logger.info("_discriminant_function(p_norm, y=0):\n"
                         "{:}".format(df_priv_scores_0))
        _check_df_scores(df_priv_scores_0, 1)

        df_priv_scores_1 = self.knn._discriminant_function(p_norm, y=1)
        self.logger.info("_discriminant_function(p_norm, y=1):\n"
                         "{:}".format(df_priv_scores_1))
        _check_df_scores(df_priv_scores_1, 1)

        df_priv_scores_2 = self.knn._discriminant_function(p_norm, y=2)
        self.logger.info("_discriminant_function(p_norm, y=2):\n"
                         "{:}".format(df_priv_scores_2))
        _check_df_scores(df_priv_scores_2, 1)

        # Comparing output of public and private

        self.assertFalse((df_scores_0 != df_priv_scores_0).any())
        self.assertFalse((df_scores_1 != df_priv_scores_1).any())
        self.assertFalse((df_scores_2 != df_priv_scores_2).any())

        self.logger.info("Testing classify on single point")

        labels, scores = self.knn.classify(p)
        self.logger.info(
            "classify(p):\nlabels: {:}\nscores: {:}".format(labels, scores))
        _check_classify_scores(labels, scores, 1, self.knn.n_classes)

        # Comparing output of discriminant_function and classify

        self.assertFalse(
            (df_scores_0 != CArray(scores[:, 0]).ravel()).any())
        self.assertFalse(
            (df_scores_1 != CArray(scores[:, 1]).ravel()).any())
        self.assertFalse(
            (df_scores_2 != CArray(scores[:, 2]).ravel()).any())


if __name__ == '__main__':
    CUnitTest.main()
