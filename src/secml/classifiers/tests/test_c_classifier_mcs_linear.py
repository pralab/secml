from secml.utils import CUnitTest

from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier

from secml.array import CArray
from secml.figure import CFigure
from secml.data.loader import CDLRandom
from secml.classifiers import CClassifierMCSLinear, CClassifierSVM
from secml.peval.metrics import CMetric


class TestCClassifierMCSLinear(CUnitTest):
    """Unit test for CClassifierMCSLinear."""

    def setUp(self):
        self.dataset = CDLRandom(n_samples=1000, n_features=500,
                                 n_redundant=0, n_informative=50,
                                 n_clusters_per_class=1).load()

    def test_classification(self):

        with self.timer():
            self.mcs = CClassifierMCSLinear(CClassifierSVM(),
                                            num_classifiers=10,
                                            max_features=0.5,
                                            max_samples=0.5)
            self.mcs.train(self.dataset)
            self.logger.info("Trained MCS.")

        with self.timer():
            self.sklearn_bagging = BaggingClassifier(SVC(),
                                                     n_estimators=10,
                                                     max_samples=0.5,
                                                     max_features=0.5,
                                                     bootstrap=False)
            self.sklearn_bagging.fit(self.dataset.X.get_data(),
                                     self.dataset.Y.tondarray())
            self.logger.info("Trained Sklearn Bagging + SVC.")

        label_mcs, s_mcs = self.mcs.classify(self.dataset.X)
        label_skbag = self.sklearn_bagging.predict(self.dataset.X.get_data())

        f1_mcs = CMetric.create('f1').performance_score(
            self.dataset.Y, label_mcs)
        f1_skbag = CMetric.create('f1').performance_score(
            self.dataset.Y, CArray(label_skbag))

        self.logger.info("F1-Score of MCS: {:}".format(f1_mcs))
        self.logger.info(
            "F1-Score of Sklearn Bagging + SVC: {:}".format(f1_skbag))
        self.assertLess(
            abs(f1_mcs - f1_skbag), 0.1,
            "Performance difference is: {:}".format(abs(f1_mcs - f1_skbag)))

    def test_plot(self):

        self.dataset = CDLRandom(n_features=2, n_redundant=0, n_informative=1,
                                 n_clusters_per_class=1).load()

        self.logger.info("Training MCS on 2D Dataset... ")
        self.mcs = CClassifierMCSLinear(CClassifierSVM(),
                                        max_features=0.5, max_samples=0.5)
        self.mcs.train(self.dataset)

        fig = CFigure()
        # Plot dataset points
        fig.switch_sptype(sp_type='ds')
        fig.sp.plot_ds(self.dataset)
        # Plot objective function
        fig.switch_sptype(sp_type='function')
        fig.sp.plot_fobj(self.mcs.discriminant_function,
                         grid_limits=self.dataset.get_bounds())
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

        mcs = CClassifierMCSLinear(
            CClassifierSVM(), num_classifiers=10,
            max_features=0.5, max_samples=0.5)
        mcs.train(self.dataset)

        # Testing discriminant_function on multiple points

        df_scores_pos = mcs.discriminant_function(
            self.dataset.X, label=1)
        self.logger.info("discriminant_function("
                         "dataset.X, label=1:\n{:}".format(df_scores_pos))
        _check_df_scores(df_scores_pos, self.dataset.num_samples)

        df_scores_neg = mcs.discriminant_function(
            self.dataset.X, label=0)
        self.logger.info("discriminant_function("
                         "dataset.X, label=0:\n{:}".format(df_scores_neg))
        _check_df_scores(df_scores_neg, self.dataset.num_samples)

        self.assertFalse(
            ((df_scores_pos.sign() * -1) != df_scores_neg.sign()).any())

        # Testing _discriminant_function on multiple points

        ds_priv_scores = mcs._discriminant_function(
            self.dataset.X, label=1)
        self.logger.info("_discriminant_function("
                         "dataset.X, label=1:\n{:}".format(ds_priv_scores))
        _check_df_scores(ds_priv_scores, self.dataset.num_samples)

        # Comparing output of public and private

        self.assertFalse((df_scores_pos != ds_priv_scores).any())

        # Testing classify on multiple points

        labels, scores = mcs.classify(self.dataset.X)
        self.logger.info("classify(dataset.X:"
                         "\nlabels: {:}\nscores:{:}".format(labels,
                                                            scores))
        _check_classify_scores(
            labels, scores, self.dataset.num_samples, mcs.n_classes)

        # Comparing output of discriminant_function and classify

        self.assertFalse((df_scores_neg != scores[:, 0].ravel()).any())
        self.assertFalse((df_scores_pos != scores[:, 1].ravel()).any())

        # Testing discriminant_function on single point

        df_scores_pos = mcs.discriminant_function(
            self.dataset.X[0, :].ravel(), label=1)
        self.logger.info("discriminant_function(dataset.X[0, :].ravel(), "
                         "label=1:\n{:}".format(df_scores_pos))
        _check_df_scores(df_scores_pos, 1)

        df_scores_neg = mcs.discriminant_function(
            self.dataset.X[0, :].ravel(), label=0)
        self.logger.info("discriminant_function(dataset.X[0, :].ravel(), "
                         "label=0:\n{:}".format(df_scores_neg))
        _check_df_scores(df_scores_neg, 1)

        self.assertFalse(
            ((df_scores_pos.sign() * -1) != df_scores_neg.sign()).any())

        # Testing _discriminant_function on single point

        df_priv_scores = mcs._discriminant_function(
            self.dataset.X[0, :].ravel(), label=1)
        self.logger.info("_discriminant_function(dataset.X[0, :].ravel(), "
                         "label=1:\n{:}".format(df_priv_scores))
        _check_df_scores(df_priv_scores, 1)

        # Comparing output of public and private

        self.assertFalse((df_scores_pos != df_priv_scores).any())

        # Testing error raising

        with self.assertRaises(ValueError):
            mcs._discriminant_function(self.dataset.X, label=0)
        with self.assertRaises(ValueError):
            mcs._discriminant_function(
                self.dataset.X[0, :].ravel(), label=0)

        self.logger.info("Testing classify on single point")

        labels, scores = mcs.classify(self.dataset.X[0, :].ravel())
        self.logger.info("classify(self.dataset.X[0, :].ravel():"
                         "\nlabels: {:}\nscores:{:}".format(labels,
                                                            scores))
        _check_classify_scores(labels, scores, 1, mcs.n_classes)

        # Comparing output of discriminant_function and classify

        self.assertFalse(
            (df_scores_neg != CArray(scores[:, 0]).ravel()).any())
        self.assertFalse(
            (df_scores_pos != CArray(scores[:, 1]).ravel()).any())


if __name__ == '__main__':
    CUnitTest.main()
