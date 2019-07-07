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
        fig.switch_sptype(sp_type='ds')
        fig.sp.plot_ds(self.dataset)

        self.log.fit(self.dataset)

        fig.switch_sptype(sp_type='function')
        fig.sp.plot_fobj(self.log.decision_function, y=1)
        fig.title('Logistic Classifier')

        self.logger.info(self.log.predict(self.dataset.X))

        fig.show()

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

        self.log.fit(self.dataset)

        x = x_norm = self.dataset.X
        p = p_norm = self.dataset.X[0, :].ravel()

        # Preprocessing data if a preprocess is defined
        if self.log.preprocess is not None:
            x_norm = self.log.preprocess.transform(x)
            p_norm = self.log.preprocess.transform(p)

        # Testing decision_function on multiple points

        df_scores_neg = self.log.decision_function(x, y=0)
        self.logger.info("decision_function(x, y=0):\n"
                         "{:}".format(df_scores_neg))
        _check_df_scores(df_scores_neg, self.dataset.num_samples)

        df_scores_pos = self.log.decision_function(x, y=1)
        self.logger.info("decision_function(x, y=1):\n"
                         "{:}".format(df_scores_pos))
        _check_df_scores(df_scores_pos, self.dataset.num_samples)

        self.assertFalse(
            ((df_scores_pos.sign() * -1) != df_scores_neg.sign()).any())

        # Testing _decision_function on multiple points

        ds_priv_scores = self.log._decision_function(x_norm, y=1)
        self.logger.info("_decision_function(x_norm, y=1):\n"
                         "{:}".format(ds_priv_scores))
        _check_df_scores(ds_priv_scores, self.dataset.num_samples)

        # Comparing output of public and private

        self.assertFalse((df_scores_pos != ds_priv_scores).any())

        # Testing predict on multiple points

        labels, scores = self.log.predict(x, return_decision_function=True)
        self.logger.info("predict(x):\nlabels: {:}\n"
                         "scores: {:}".format(labels, scores))
        _check_classify_scores(
            labels, scores, self.dataset.num_samples, self.log.n_classes)

        # Comparing output of decision_function and predict

        self.assertFalse((df_scores_neg != scores[:, 0].ravel()).any())
        self.assertFalse((df_scores_pos != scores[:, 1].ravel()).any())

        # Testing decision_function on single point

        df_scores_neg = self.log.decision_function(p, y=0)
        self.logger.info("decision_function(p, y=0):\n"
                         "{:}".format(df_scores_neg))
        _check_df_scores(df_scores_neg, 1)

        df_scores_pos = self.log.decision_function(p, y=1)
        self.logger.info("decision_function(p, y=1):\n"
                         "{:}".format(df_scores_pos))
        _check_df_scores(df_scores_pos, 1)

        self.assertFalse(
            ((df_scores_pos.sign() * -1) != df_scores_neg.sign()).any())

        # Testing _decision_function on single point

        df_priv_scores = self.log._decision_function(p_norm, y=1)
        self.logger.info("_decision_function(p_norm, y=1):\n"
                         "{:}".format(df_priv_scores))
        _check_df_scores(df_priv_scores, 1)

        # Comparing output of public and private

        self.assertFalse((df_scores_pos != df_priv_scores).any())

        self.logger.info("Testing predict on single point")

        labels, scores = self.log.predict(p, return_decision_function=True)
        self.logger.info("predict(p):\nlabels: {:}\n"
                         "scores: {:}".format(labels, scores))
        _check_classify_scores(labels, scores, 1, self.log.n_classes)

        # Comparing output of decision_function and predict

        self.assertFalse(
            (df_scores_neg != CArray(scores[:, 0]).ravel()).any())
        self.assertFalse(
            (df_scores_pos != CArray(scores[:, 1]).ravel()).any())

        # Testing error raising

        with self.assertRaises(ValueError):
            self.log._decision_function(x_norm, y=0)
        with self.assertRaises(ValueError):
            self.log._decision_function(p_norm, y=0)

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
