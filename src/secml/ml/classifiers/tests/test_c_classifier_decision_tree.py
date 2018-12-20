from secml.utils import CUnitTest

from secml.data.loader import CDLRandomToy
from secml.ml.classifiers import CClassifierDecisionTree
from secml.array import CArray


class TestCClassifierDecisionTree(CUnitTest):
    """Unit test for CDecisionTree."""

    def setUp(self):
        self.dataset = CDLRandomToy('iris').load()

        self.dec_tree = CClassifierDecisionTree()

    def test_classify(self):
        """Test for predict method. """
        self.logger.info("Testing decision tree classifier training ")
        self.dec_tree.fit(self.dataset)

        self.logger.info("Testing classification with trees")

        self.logger.info(
            "Number of classes: {:}".format(self.dec_tree.n_classes))

        y, result = self.dec_tree.predict(
            self.dataset.X[0, :], return_decision_function=True)
        self.logger.info(
            "Probability of affinity to each class: {:}".format(result))
        self.logger.info("Class of affinity: {:}".format(y))
        self.assertEquals(y, self.dataset.Y[0], "Wrong classification")

        y, result = self.dec_tree.predict(
            self.dataset.X[50, :], return_decision_function=True)
        self.logger.info(
            "Probability of affinity to each class: {:}".format(result))
        self.logger.info("Class of affinity: {:}".format(y))
        self.assertEquals(y, self.dataset.Y[50], "Wrong classification")

        y, result = self.dec_tree.predict(
            self.dataset.X[120, :], return_decision_function=True)
        self.logger.info(
            "Probability of affinity to each class: {:}".format(result))
        self.logger.info("Class of affinity: {:}".format(y))
        self.assertEquals(y, self.dataset.Y[120], "Wrong classification")

    def test_fun(self):
        """Test for decision_function() and predict() methods."""
        self.logger.info(
            "Test for decision_function() and predict() methods.")

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

        self.dec_tree.fit(self.dataset)

        x = x_norm = self.dataset.X
        p = p_norm = self.dataset.X[0, :].ravel()

        # Normalizing data if a normalizer is defined
        if self.dec_tree.normalizer is not None:
            x_norm = self.dec_tree.normalizer.normalize(x)
            p_norm = self.dec_tree.normalizer.normalize(p)

        # Testing decision_function on multiple points

        df_scores_0 = self.dec_tree.decision_function(x, y=0)
        self.logger.info(
            "decision_function(x, y=0):\n{:}".format(df_scores_0))
        _check_df_scores(df_scores_0, self.dataset.num_samples)

        df_scores_1 = self.dec_tree.decision_function(x, y=1)
        self.logger.info(
            "decision_function(x, y=1):\n{:}".format(df_scores_1))
        _check_df_scores(df_scores_1, self.dataset.num_samples)

        df_scores_2 = self.dec_tree.decision_function(x, y=2)
        self.logger.info(
            "decision_function(x, y=2):\n{:}".format(df_scores_2))
        _check_df_scores(df_scores_2, self.dataset.num_samples)

        # Testing _decision_function on multiple points

        ds_priv_scores_0 = self.dec_tree._decision_function(x_norm, y=0)
        self.logger.info("_decision_function(x_norm, y=0):\n"
                         "{:}".format(ds_priv_scores_0))
        _check_df_scores(ds_priv_scores_0, self.dataset.num_samples)

        ds_priv_scores_1 = self.dec_tree._decision_function(x_norm, y=1)
        self.logger.info("_decision_function(x_norm, y=1):\n"
                         "{:}".format(ds_priv_scores_1))
        _check_df_scores(ds_priv_scores_1, self.dataset.num_samples)

        ds_priv_scores_2 = self.dec_tree._decision_function(x_norm, y=2)
        self.logger.info("_decision_function(x_norm, y=2):\n"
                         "{:}".format(ds_priv_scores_2))
        _check_df_scores(ds_priv_scores_2, self.dataset.num_samples)

        # Comparing output of public and private

        self.assertFalse((df_scores_0 != ds_priv_scores_0).any())
        self.assertFalse((df_scores_1 != ds_priv_scores_1).any())
        self.assertFalse((df_scores_2 != ds_priv_scores_2).any())

        # Testing predict on multiple points

        labels, scores = self.dec_tree.predict(
            x, return_decision_function=True)
        self.logger.info(
            "predict(x):\nlabels: {:}\nscores:{:}".format(labels, scores))
        _check_classify_scores(
            labels, scores, self.dataset.num_samples, self.dec_tree.n_classes)

        # Comparing output of decision_function and predict

        self.assertFalse((df_scores_0 != scores[:, 0].ravel()).any())
        self.assertFalse((df_scores_1 != scores[:, 1].ravel()).any())
        self.assertFalse((df_scores_2 != scores[:, 2].ravel()).any())

        # Testing decision_function on single point

        df_scores_0 = self.dec_tree.decision_function(p, y=0)
        self.logger.info(
            "decision_function(p, y=0):\n{:}".format(df_scores_0))
        _check_df_scores(df_scores_0, 1)

        df_scores_1 = self.dec_tree.decision_function(p, y=1)
        self.logger.info(
            "decision_function(p, y=1):\n{:}".format(df_scores_1))
        _check_df_scores(df_scores_1, 1)

        df_scores_2 = self.dec_tree.decision_function(p, y=2)
        self.logger.info(
            "decision_function(p, y=2):\n{:}".format(df_scores_2))
        _check_df_scores(df_scores_2, 1)

        # Testing _decision_function on single point

        df_priv_scores_0 = self.dec_tree._decision_function(p_norm, y=0)
        self.logger.info("_decision_function(p_norm, y=0):\n"
                         "{:}".format(df_priv_scores_0))
        _check_df_scores(df_priv_scores_0, 1)

        df_priv_scores_1 = self.dec_tree._decision_function(p_norm, y=1)
        self.logger.info("_decision_function(p_norm, y=1):\n"
                         "{:}".format(df_priv_scores_1))
        _check_df_scores(df_priv_scores_1, 1)

        df_priv_scores_2 = self.dec_tree._decision_function(p_norm, y=2)
        self.logger.info("_decision_function(p_norm, y=2):\n"
                         "{:}".format(df_priv_scores_2))
        _check_df_scores(df_priv_scores_2, 1)

        # Comparing output of public and private

        self.assertFalse((df_scores_0 != df_priv_scores_0).any())
        self.assertFalse((df_scores_1 != df_priv_scores_1).any())
        self.assertFalse((df_scores_2 != df_priv_scores_2).any())

        self.logger.info("Testing predict on single point")

        labels, scores = self.dec_tree.predict(
            p, return_decision_function=True)
        self.logger.info(
            "predict(p):\nlabels: {:}\nscores: {:}".format(labels, scores))
        _check_classify_scores(labels, scores, 1, self.dec_tree.n_classes)

        # Comparing output of decision_function and predict

        self.assertFalse(
            (df_scores_0 != CArray(scores[:, 0]).ravel()).any())
        self.assertFalse(
            (df_scores_1 != CArray(scores[:, 1]).ravel()).any())
        self.assertFalse(
            (df_scores_2 != CArray(scores[:, 2]).ravel()).any())


if __name__ == '__main__':
    CUnitTest.main()
