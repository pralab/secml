from secml.utils import CUnitTest

from secml.data.loader import CDLRandomToy
from secml.classifiers import CClassifierRandomForest
from secml.array import CArray


class TestCClassifierRandomForest(CUnitTest):
    """Unit test for CRandomForest."""

    def setUp(self):
        self.dataset = CDLRandomToy('iris').load()

        self.rnd_forest = CClassifierRandomForest()
       
    def test_classify(self):

        self.logger.info("Testing random forest training ")
        self.rnd_forest.train(self.dataset)

        self.logger.info("Testing classification with trees")
        
        self.logger.info(
            "Number of classes: {:}".format(self.rnd_forest.n_classes))
        
        y, result = self.rnd_forest.classify(self.dataset.X[0, :])
        self.logger.info(
            "Probability of affinity to each class: {:}".format(result))
        self.logger.info("Class of affinity: {:}".format(y))
        self.assertEquals(y, self.dataset.Y[0], "Wrong classification")
        
        y, result = self.rnd_forest.classify(self.dataset.X[50, :])
        self.logger.info(
            "Probability of affinity to each class: {:}".format(result))
        self.logger.info("Class of affinity: {:}".format(y))
        self.assertEquals(y, self.dataset.Y[50], "Wrong classification")
        
        y, result = self.rnd_forest.classify(self.dataset.X[120, :])
        self.logger.info(
            "Probability of affinity to each class: {:}".format(result))
        self.logger.info("Class of affinity: {:}".format(y))
        self.assertEquals(y, self.dataset.Y[120], "Wrong classification")

    def test_fun(self):
        """Test for discriminant_function() and classify() methods."""
        self.logger.info(
            "Test for discriminant_function() and classify() methods.")

        def _check_df_scores(s, n_samples):
            self.assertEqual(type(s), CArray)
            self.assertTrue(s.isdense)
            self.assertEqual(1, s.ndim)
            self.assertEqual((n_samples,), df_scores_0.shape)
            self.assertEqual(df_scores_0.dtype, float)

        def _check_classify_scores(l, s, n_samples, n_classes):
            self.assertEqual(type(l), CArray)
            self.assertEqual(type(s), CArray)
            self.assertTrue(l.isdense)
            self.assertTrue(s.isdense)
            self.assertEqual(1, l.ndim)
            self.assertEqual(2, s.ndim)
            self.assertEqual((n_samples,), l.shape)
            self.assertEqual((n_samples, n_classes), s.shape)
            self.assertEqual(l.dtype, int)
            self.assertEqual(s.dtype, float)

        self.rnd_forest.train(self.dataset)

        # Testing discriminant_function on multiple points

        df_scores_0 = self.rnd_forest.discriminant_function(
            self.dataset.X, label=0)
        self.logger.info("discriminant_function("
                         "dataset.X, label=0:\n{:}".format(df_scores_0))
        _check_df_scores(df_scores_0, self.dataset.num_samples)

        df_scores_1 = self.rnd_forest.discriminant_function(
            self.dataset.X, label=1)
        self.logger.info("discriminant_function("
                         "dataset.X, label=1:\n{:}".format(df_scores_1))
        _check_df_scores(df_scores_1, self.dataset.num_samples)

        df_scores_2 = self.rnd_forest.discriminant_function(
            self.dataset.X, label=2)
        self.logger.info("discriminant_function("
                         "dataset.X, label=2:\n{:}".format(df_scores_2))
        _check_df_scores(df_scores_2, self.dataset.num_samples)

        # Testing _discriminant_function on multiple points

        ds_priv_scores_0 = self.rnd_forest._discriminant_function(
            self.dataset.X, label=0)
        self.logger.info("_discriminant_function("
                         "dataset.X, label=0:\n{:}".format(ds_priv_scores_0))
        _check_df_scores(ds_priv_scores_0, self.dataset.num_samples)

        ds_priv_scores_1 = self.rnd_forest._discriminant_function(
            self.dataset.X, label=1)
        self.logger.info("_discriminant_function("
                         "dataset.X, label=1:\n{:}".format(ds_priv_scores_1))
        _check_df_scores(ds_priv_scores_1, self.dataset.num_samples)

        ds_priv_scores_2 = self.rnd_forest._discriminant_function(
            self.dataset.X, label=2)
        self.logger.info("_discriminant_function("
                         "dataset.X, label=2:\n{:}".format(ds_priv_scores_2))
        _check_df_scores(ds_priv_scores_2, self.dataset.num_samples)

        # Comparing output of public and private

        self.assertFalse((df_scores_0 != ds_priv_scores_0).any())
        self.assertFalse((df_scores_1 != ds_priv_scores_1).any())
        self.assertFalse((df_scores_2 != ds_priv_scores_2).any())

        # Testing classify on multiple points

        labels, scores = self.rnd_forest.classify(self.dataset.X)
        self.logger.info("classify(dataset.X:\nlabels: {:}"
                         "\nscores:{:}".format(labels, scores))
        _check_classify_scores(
            labels, scores, self.dataset.num_samples, self.rnd_forest.n_classes)

        # Comparing output of discriminant_function and classify

        self.assertFalse((df_scores_0 != scores[:, 0].ravel()).any())
        self.assertFalse((df_scores_1 != scores[:, 1].ravel()).any())
        self.assertFalse((df_scores_2 != scores[:, 2].ravel()).any())

        # Testing discriminant_function on single point

        df_scores_0 = self.rnd_forest.discriminant_function(
            self.dataset.X[0, :].ravel(), label=0)
        self.logger.info("discriminant_function(dataset.X[0, :].ravel(), "
                         "label=0:\n{:}".format(df_scores_0))
        _check_df_scores(df_scores_0, 1)

        df_scores_1 = self.rnd_forest.discriminant_function(
            self.dataset.X[0, :].ravel(), label=1)
        self.logger.info("discriminant_function(dataset.X[0, :].ravel(), "
                         "label=1:\n{:}".format(df_scores_1))
        _check_df_scores(df_scores_1, 1)

        df_scores_2 = self.rnd_forest.discriminant_function(
            self.dataset.X[0, :].ravel(), label=2)
        self.logger.info("discriminant_function(dataset.X[0, :].ravel(), "
                         "label=2:\n{:}".format(df_scores_2))
        _check_df_scores(df_scores_2, 1)

        # Testing _discriminant_function on single point

        df_priv_scores_0 = self.rnd_forest._discriminant_function(
            self.dataset.X[0, :].ravel(), label=0)
        self.logger.info("_discriminant_function(dataset.X[0, :].ravel(), "
                         "label=0:\n{:}".format(df_priv_scores_0))
        _check_df_scores(df_priv_scores_0, 1)

        df_priv_scores_1 = self.rnd_forest._discriminant_function(
            self.dataset.X[0, :].ravel(), label=1)
        self.logger.info("_discriminant_function(dataset.X[0, :].ravel(), "
                         "label=1:\n{:}".format(df_priv_scores_1))
        _check_df_scores(df_priv_scores_1, 1)

        df_priv_scores_2 = self.rnd_forest._discriminant_function(
            self.dataset.X[0, :].ravel(), label=2)
        self.logger.info("_discriminant_function(dataset.X[0, :].ravel(), "
                         "label=2:\n{:}".format(df_priv_scores_2))
        _check_df_scores(df_priv_scores_2, 1)

        # Comparing output of public and private

        self.assertFalse((df_scores_0 != df_priv_scores_0).any())
        self.assertFalse((df_scores_1 != df_priv_scores_1).any())
        self.assertFalse((df_scores_2 != df_priv_scores_2).any())

        self.logger.info("Testing classify on single point")

        labels, scores = self.rnd_forest.classify(self.dataset.X[0, :].ravel())
        self.logger.info("classify(self.dataset.X[0, :].ravel():\nlabels: "
                         "{:}\nscores:{:}".format(labels, scores))
        _check_classify_scores(labels, scores, 1, self.rnd_forest.n_classes)

        # Comparing output of discriminant_function and classify

        self.assertFalse(
            (df_scores_0 != CArray(scores[:, 0]).ravel()).any())
        self.assertFalse(
            (df_scores_1 != CArray(scores[:, 1]).ravel()).any())
        self.assertFalse(
            (df_scores_2 != CArray(scores[:, 2]).ravel()).any())


if __name__ == '__main__':
    CUnitTest.main()
