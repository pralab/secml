from secml.utils import CUnitTest

from secml.data.loader import CDLRandom
from secml.classifiers import CClassifierKDE
from secml.array import CArray
from secml.kernel import CKernelRBF
from secml.features.normalization import CNormalizerMinMax
from secml.figure import CFigure


class TestCClassifierKDE(CUnitTest):
    """Unit test for CClassifierKDE."""

    def setUp(self):
        """Test for init and train methods."""
        # generate synthetic data
        self.dataset = CDLRandom(n_features=2, n_redundant=0, n_informative=1,
                                 n_clusters_per_class=1).load()

        self.dataset.X = CNormalizerMinMax().train_normalize(self.dataset.X)

        self.logger.info("Testing classifier creation ")
        k = CKernelRBF(gamma=1e1)
        # k = CKernelLinear
        self.kde = CClassifierKDE(k)

        self.logger.info("Testing Stochastic gradient descent classifier training ")

    def test_draw(self):
        """ Compare the classifiers graphically"""
        self.logger.info("Testing classifiers graphically")
        # Preparation of the grid
        fig = CFigure()
        fig.switch_sptype(sp_type='ds')
        fig.sp.plot_ds(self.dataset)

        self.kde.train(self.dataset)

        fig.switch_sptype(sp_type='function')
        fig.sp.plot_fobj(self.kde.discriminant_function, y=1)
        fig.title('kde Classifier')

        self.logger.info(self.kde.classify(self.dataset.X))

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

        self.kde.train(self.dataset)

        x = x_norm = self.dataset.X
        p = p_norm = self.dataset.X[0, :].ravel()

        # Normalizing data if a normalizer is defined
        if self.kde.normalizer is not None:
            x_norm = self.kde.normalizer.normalize(x)
            p_norm = self.kde.normalizer.normalize(p)

        # Testing discriminant_function on multiple points

        df_scores_neg = self.kde.discriminant_function(x, y=0)
        self.logger.info(
            "discriminant_function(p_norm, y=0):\n{:}".format(df_scores_neg))
        _check_df_scores(df_scores_neg, self.dataset.num_samples)

        df_scores_pos = self.kde.discriminant_function(x, y=1)
        self.logger.info(
            "discriminant_function(x, y=1):\n{:}".format(df_scores_pos))
        _check_df_scores(df_scores_pos, self.dataset.num_samples)

        self.assertFalse(((1 - df_scores_neg) != df_scores_pos).any())

        # Testing _discriminant_function on multiple points

        ds_priv_scores_neg = self.kde._discriminant_function(x_norm, y=0)
        self.logger.info("_discriminant_function(x_norm, y=0):\n"
                         "{:}".format(ds_priv_scores_neg))
        _check_df_scores(ds_priv_scores_neg, self.dataset.num_samples)

        ds_priv_scores_pos = self.kde._discriminant_function(x_norm, y=1)
        self.logger.info("_discriminant_function(x_norm, y=1):\n"
                         "{:}".format(ds_priv_scores_pos))
        _check_df_scores(ds_priv_scores_pos, self.dataset.num_samples)

        # Comparing output of public and private

        self.assertFalse((df_scores_pos != ds_priv_scores_pos).any())
        self.assertFalse((df_scores_neg != ds_priv_scores_neg).any())

        # Testing classify on multiple points

        labels, scores = self.kde.classify(x)
        self.logger.info(
            "classify(x):\nlabels: {:}\nscores: {:}".format(labels, scores))
        _check_classify_scores(
            labels, scores, self.dataset.num_samples, self.kde.n_classes)

        # Comparing output of discriminant_function and classify

        self.assertFalse((df_scores_neg != scores[:, 0].ravel()).any())
        self.assertFalse((df_scores_pos != scores[:, 1].ravel()).any())

        # Testing discriminant_function on single point

        df_scores_neg = self.kde.discriminant_function(p, y=0)
        self.logger.info(
            "discriminant_function(p, y=0):\n{:}".format(df_scores_neg))
        _check_df_scores(df_scores_neg, 1)

        df_scores_pos = self.kde.discriminant_function(p, y=1)
        self.logger.info(
            "discriminant_function(p, y=1):\n{:}".format(df_scores_pos))
        _check_df_scores(df_scores_pos, 1)

        self.assertFalse(((1 - df_scores_neg) != df_scores_pos).any())

        # Testing _discriminant_function on single point

        df_priv_scores_neg = self.kde._discriminant_function(p_norm, y=0)
        self.logger.info("_discriminant_function(p_norm, y=0):\n"
                         "{:}".format(df_priv_scores_neg))
        _check_df_scores(df_priv_scores_neg, 1)

        df_priv_scores_pos = self.kde._discriminant_function(p_norm, y=1)
        self.logger.info("_discriminant_function(p_norm, y=1):\n"
                         "{:}".format(df_priv_scores_pos))
        _check_df_scores(df_priv_scores_pos, 1)

        # Comparing output of public and private

        self.assertFalse((df_scores_pos != df_priv_scores_pos).any())
        self.assertFalse((df_scores_neg != df_priv_scores_neg).any())

        self.logger.info("Testing classify on single point")

        labels, scores = self.kde.classify(p)
        self.logger.info(
            "classify(p):\nlabels: {:}\nscores: {:}".format(labels, scores))
        _check_classify_scores(labels, scores, 1, self.kde.n_classes)

        # Comparing output of discriminant_function and classify

        self.assertFalse(
            (df_scores_neg != CArray(scores[:, 0]).ravel()).any())
        self.assertFalse(
            (df_scores_pos != CArray(scores[:, 1]).ravel()).any())


if __name__ == '__main__':
    CUnitTest.main()
