from secml.utils import CUnitTest

from secml.data.loader import CDLRandom
from secml.classifiers import CClassifierNearestCentroid
from secml.array import CArray
from secml.figure import CFigure
from secml.features.normalization import CNormalizerMinMax


class TestCClassifierNearestCentroid(CUnitTest):
    """Unit test for CClassifierNearestCentroid."""

    def setUp(self):
        """Test for init and train methods."""

        self.dataset = CDLRandom(n_features=2, n_redundant=0, n_informative=1,
                                 n_clusters_per_class=1).load()

        self.dataset.X = CNormalizerMinMax().train_normalize(self.dataset.X)

        self.nc = CClassifierNearestCentroid()

    def test_draw(self):
        """ Compare the classifiers graphically"""
        self.logger.info("Testing classifiers graphically")
        # Preparation of the grid
        fig = CFigure()
        fig.switch_sptype(sp_type='ds')
        fig.sp.plot_ds(self.dataset)

        self.nc.train(self.dataset)

        fig.switch_sptype(sp_type='function')
        fig.sp.plot_fobj(self.nc.discriminant_function, label=1)
        fig.title('nearest centroid  Classifier')

        self.logger.info(self.nc.classify(self.dataset.X))

        fig.show()

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

        self.nc.train(self.dataset)

        # Testing discriminant_function on multiple points

        df_scores_pos = self.nc.discriminant_function(
            self.dataset.X, label=1)
        self.logger.info("discriminant_function("
                         "dataset.X, label=1:\n{:}".format(df_scores_pos))
        _check_df_scores(df_scores_pos, self.dataset.num_samples)

        df_scores_neg = self.nc.discriminant_function(
            self.dataset.X, label=0)
        self.logger.info("discriminant_function("
                         "dataset.X, label=0:\n{:}".format(df_scores_neg))
        _check_df_scores(df_scores_neg, self.dataset.num_samples)

        self.assertFalse(
            ((df_scores_pos.sign() * -1) != df_scores_neg.sign()).any())

        # Testing _discriminant_function on multiple points

        ds_priv_scores = self.nc._discriminant_function(
            self.dataset.X, label=1)
        self.logger.info("_discriminant_function("
                         "dataset.X, label=1:\n{:}".format(ds_priv_scores))
        _check_df_scores(ds_priv_scores, self.dataset.num_samples)

        # Comparing output of public and private

        self.assertFalse((df_scores_pos != ds_priv_scores).any())

        # Testing classify on multiple points

        labels, scores = self.nc.classify(self.dataset.X)
        self.logger.info("classify(dataset.X:\nlabels: {:}"
                         "\nscores:{:}".format(labels, scores))
        _check_classify_scores(
            labels, scores, self.dataset.num_samples, self.nc.n_classes)

        # Comparing output of discriminant_function and classify

        self.assertFalse((df_scores_neg != scores[:, 0].ravel()).any())
        self.assertFalse((df_scores_pos != scores[:, 1].ravel()).any())

        # Testing discriminant_function on single point

        df_scores_pos = self.nc.discriminant_function(
            self.dataset.X[0, :].ravel(), label=1)
        self.logger.info("discriminant_function(dataset.X[0, :].ravel(), "
                         "label=1:\n{:}".format(df_scores_pos))
        _check_df_scores(df_scores_pos, 1)

        df_scores_neg = self.nc.discriminant_function(
            self.dataset.X[0, :].ravel(), label=0)
        self.logger.info("discriminant_function(dataset.X[0, :].ravel(), "
                         "label=0:\n{:}".format(df_scores_neg))
        _check_df_scores(df_scores_neg, 1)

        self.assertFalse(
            ((df_scores_pos.sign() * -1) != df_scores_neg.sign()).any())

        # Testing _discriminant_function on single point

        df_priv_scores = self.nc._discriminant_function(
            self.dataset.X[0, :].ravel(), label=1)
        self.logger.info("_discriminant_function(dataset.X[0, :].ravel(), "
                         "label=1:\n{:}".format(df_priv_scores))
        _check_df_scores(df_priv_scores, 1)

        # Comparing output of public and private

        self.assertFalse((df_scores_pos != df_priv_scores).any())

        # Testing error raising

        with self.assertRaises(ValueError):
            self.nc._discriminant_function(self.dataset.X, label=0)
        with self.assertRaises(ValueError):
            self.nc._discriminant_function(
                self.dataset.X[0, :].ravel(), label=0)

        self.logger.info("Testing classify on single point")

        labels, scores = self.nc.classify(self.dataset.X[0, :].ravel())
        self.logger.info("classify(self.dataset.X[0, :].ravel():\nlabels: "
                         "{:}\nscores:{:}".format(labels, scores))
        _check_classify_scores(labels, scores, 1, self.nc.n_classes)

        # Comparing output of discriminant_function and classify

        self.assertFalse(
            (df_scores_neg != CArray(scores[:, 0]).ravel()).any())
        self.assertFalse(
            (df_scores_pos != CArray(scores[:, 1]).ravel()).any())


if __name__ == '__main__':
    CUnitTest.main()
