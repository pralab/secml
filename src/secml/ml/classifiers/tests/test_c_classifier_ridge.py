from secml.utils import CUnitTest

from secml.array import CArray
from secml.ml.classifiers import CClassifierRidge, CClassifierSVM
from secml.ml.kernel import *
from secml.data.loader import CDLRandom
from secml.ml.features.normalization import CNormalizerMinMax
from secml.ml.peval.metrics import CMetric
from secml.figure import CFigure


class TestCClassifierRidge(CUnitTest):
    """Unit test for Ridge Classifier."""

    def setUp(self):
        """Test for init and fit methods."""
        # generate synthetic data
        self.dataset = CDLRandom(n_features=1000, n_redundant=200,
                                 n_informative=250,
                                 n_clusters_per_class=2).load()

        self.dataset.X = CNormalizerMinMax().fit_normalize(self.dataset.X)

        kernel_types = (None, CKernelLinear, CKernelRBF, CKernelPoly)
        self.ridges = [CClassifierRidge(
            alpha=1e-6, kernel=kernel() if kernel is not None else None)
                for kernel in kernel_types]
        self.logger.info(
            "Testing RIDGE with kernel unctions: %s", str(kernel_types))

        for ridge in self.ridges:
            ridge.verbose = 2  # Enabling debug output for each classifier
            ridge.fit(self.dataset)

    def test_time(self):
        """ Compare execution time of ridge and SVM"""
        self.logger.info("Testing training speed of ridge compared to SVM ")

        for ridge in self.ridges:

            self.logger.info("RIDGE kernel: {:}".format(ridge.kernel))

            svm = CClassifierSVM(ridge.kernel)

            with self.timer() as t_svm:
                svm.fit(self.dataset)
            self.logger.info(
                "Execution time of SVM: {:}".format(t_svm.interval))
            with self.timer() as t_ridge:
                ridge.fit(self.dataset)
            self.logger.info(
                "Execution time of ridge: {:}".format(t_ridge.interval))

    def test_draw(self):
        """ Compare the classifiers graphically"""
        self.logger.info("Testing classifiers graphically")

        # generate 2D synthetic data
        dataset = CDLRandom(n_features=2, n_redundant=0, n_informative=2,
                            n_clusters_per_class=1).load()
        dataset.X = CNormalizerMinMax().fit_normalize(dataset.X)

        self.ridges[0].fit(dataset)

        svm = CClassifierSVM()
        svm.fit(dataset)

        fig = CFigure(width=10, markersize=8)
        fig.subplot(2, 1, 1, sp_type='ds')
        # Plot dataset points
        fig.sp.plot_ds(dataset)
        # Plot objective function
        fig.switch_sptype(sp_type='function')
        fig.sp.plot_fobj(svm.decision_function,
                         grid_limits=dataset.get_bounds())
        fig.sp.title('SVM')

        fig.subplot(2, 1, 2, sp_type='ds')
        # Plot dataset points
        fig.sp.plot_ds(dataset)
        # Plot objective function
        fig.switch_sptype(sp_type='function')
        fig.sp.plot_fobj(self.ridges[0].decision_function,
                         grid_limits=dataset.get_bounds())
        fig.sp.title('ridge Classifier')

        fig.show()

    def test_performance(self):
        """ Compare the classifiers performance"""
        self.logger.info("Testing error performance of the "
                         "classifiers on the training set")

        for ridge in self.ridges:

            self.logger.info("RIDGE kernel: {:}".format(ridge.kernel))

            svm = CClassifierSVM(ridge.kernel)
            svm.fit(self.dataset)

            label_svm, y_svm = svm.predict(
                self.dataset.X, return_decision_function=True)
            label_ridge, y_ridge = ridge.predict(
                self.dataset.X, return_decision_function=True)

            acc_svm = CMetric.create('f1').performance_score(
                self.dataset.Y, label_svm)
            acc_ridge = CMetric.create('f1').performance_score(
                self.dataset.Y, label_ridge)

            self.logger.info("Accuracy of SVM: {:}".format(acc_svm))
            self.assertGreater(acc_svm, 0.90,
                               "Accuracy of SVM: {:}".format(acc_svm))
            self.logger.info("Accuracy of ridge: {:}".format(acc_ridge))
            self.assertGreater(acc_ridge, 0.90,
                               "Accuracy of ridge: {:}".format(acc_ridge))

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

        for ridge in self.ridges:

            self.logger.info("RIDGE kernel: {:}".format(ridge.kernel))

            ridge.fit(self.dataset)

            x = x_norm = self.dataset.X
            p = p_norm = self.dataset.X[0, :].ravel()

            # Preprocessing data if a preprocess is defined
            if ridge.preprocess is not None:
                x_norm = ridge.preprocess.normalize(x)
                p_norm = ridge.preprocess.normalize(p)

            # Testing decision_function on multiple points

            df_scores_neg = ridge.decision_function(x, y=0)
            self.logger.info("decision_function(x, y=0):\n"
                             "{:}".format(df_scores_neg))
            _check_df_scores(df_scores_neg, self.dataset.num_samples)

            df_scores_pos = ridge.decision_function(x, y=1)
            self.logger.info("decision_function(x, y=1):\n"
                             "{:}".format(df_scores_pos))
            _check_df_scores(df_scores_pos, self.dataset.num_samples)

            self.assertFalse(
                ((df_scores_pos.sign() * -1) != df_scores_neg.sign()).any())

            # Testing _decision_function on multiple points

            ds_priv_scores = ridge._decision_function(x_norm, y=1)
            self.logger.info("_decision_function(x_norm, y=1):\n"
                             "{:}".format(ds_priv_scores))
            _check_df_scores(ds_priv_scores, self.dataset.num_samples)

            # Comparing output of public and private

            self.assertFalse((df_scores_pos != ds_priv_scores).any())

            # Testing predict on multiple points

            labels, scores = ridge.predict(x, return_decision_function=True)
            self.logger.info("predict(x):\nlabels: {:}\n"
                             "scores: {:}".format(labels, scores))
            _check_classify_scores(
                labels, scores, self.dataset.num_samples, ridge.n_classes)

            # Comparing output of decision_function and predict

            self.assertFalse((df_scores_neg != scores[:, 0].ravel()).any())
            self.assertFalse((df_scores_pos != scores[:, 1].ravel()).any())

            # Testing decision_function on single point

            df_scores_neg = ridge.decision_function(p, y=0)
            self.logger.info("decision_function(p, y=0):\n"
                             "{:}".format(df_scores_neg))
            _check_df_scores(df_scores_neg, 1)

            df_scores_pos = ridge.decision_function(p, y=1)
            self.logger.info("decision_function(p, y=1):\n"
                             "{:}".format(df_scores_pos))
            _check_df_scores(df_scores_pos, 1)

            self.assertFalse(
                ((df_scores_pos.sign() * -1) != df_scores_neg.sign()).any())

            # Testing _decision_function on single point

            df_priv_scores = ridge._decision_function(p_norm, y=1)
            self.logger.info("_decision_function(p_norm, y=1):\n"
                             "{:}".format(df_priv_scores))
            _check_df_scores(df_priv_scores, 1)

            # Comparing output of public and private

            self.assertFalse((df_scores_pos != df_priv_scores).any())

            self.logger.info("Testing predict on single point")

            labels, scores = ridge.predict(p, return_decision_function=True)
            self.logger.info("predict(p):\nlabels: {:}\n"
                             "scores: {:}".format(labels, scores))
            _check_classify_scores(labels, scores, 1, ridge.n_classes)

            # Comparing output of decision_function and predict

            self.assertFalse(
                (df_scores_neg != CArray(scores[:, 0]).ravel()).any())
            self.assertFalse(
                (df_scores_pos != CArray(scores[:, 1]).ravel()).any())

            # Testing error raising

            with self.assertRaises(ValueError):
                ridge._decision_function(x_norm, y=0)
            with self.assertRaises(ValueError):
                ridge._decision_function(p_norm, y=0)


if __name__ == '__main__':
    CUnitTest.main()
