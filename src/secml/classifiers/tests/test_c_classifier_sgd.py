from secml.utils import CUnitTest

import numpy as np

from secml.classifiers import CClassifierSGD, CClassifierSVM
from secml.classifiers.regularizer import *
from secml.classifiers.loss import *
from secml.kernel import *
from secml.array import CArray
from secml.data.loader import CDLRandom, CDLRandomBlobs
from secml.features.normalization import CNormalizerMinMax
from secml.peval.metrics import CMetric
from secml.figure import CFigure


class TestCClassifierSGD(CUnitTest):
    """Unit test for SGD Classifier."""

    def setUp(self):
        """Test for init and train methods."""        
        # generate synthetic data
        self.dataset = CDLRandom(n_features=1000, n_redundant=200,
                                 n_informative=250,
                                 n_clusters_per_class=2).load()

        self.dataset.X = CNormalizerMinMax().train_normalize(self.dataset.X)

        self.logger.info("Testing classifier creation ")
        self.sgd = CClassifierSGD(regularizer=CRegularizerL2(),
                                  loss=CLossHinge())

        kernel_types = (None, CKernelLinear, CKernelRBF, CKernelPoly)
        self.sgds = [CClassifierSGD(
            regularizer=CRegularizerL2(), loss=CLossHinge(),
            kernel=kernel() if kernel is not None else None)
                for kernel in kernel_types]
        self.logger.info(
            "Testing SGD with kernel unctions: %s", str(kernel_types))

        for sgd in self.sgds:
            sgd.verbose = 2  # Enabling debug output for each classifier
            sgd.train(self.dataset)

    def test_time(self):
        """ Compare execution time of SGD and SVM"""
        self.logger.info("Testing training speed of SGD compared to SVM ")

        for sgd in self.sgds:

            self.logger.info("SGD kernel: {:}".format(sgd.kernel))

            svm = CClassifierSVM(sgd.kernel)

            with self.timer() as t_svm:
                svm.train(self.dataset)
            self.logger.info("Execution time of SVM: " + str(t_svm.interval) + "\n")
            with self.timer() as t_sgd:
                sgd.train(self.dataset)
            self.logger.info("Execution time of SGD: " + str(t_sgd.interval) + "\n")

    def test_draw(self):
        """ Compare the classifiers graphically"""
        self.logger.info("Testing classifiers graphically")

        # generate 2D synthetic data
        dataset = CDLRandom(n_features=2, n_redundant=1, n_informative=1,
                            n_clusters_per_class=1).load()
        dataset.X = CNormalizerMinMax().train_normalize(dataset.X)

        self.sgds[0].train(dataset)

        svm = CClassifierSVM()
        svm.train(dataset)

        fig = CFigure(width=10, markersize=8)
        fig.subplot(2, 1, 1, sp_type='ds')
        # Plot dataset points
        fig.sp.plot_ds(dataset)
        # Plot objective function
        fig.switch_sptype(sp_type='function')
        fig.sp.plot_fobj(svm.discriminant_function,
                         grid_limits=dataset.get_bounds())
        fig.sp.title('SVM')

        fig.subplot(2, 1, 2, sp_type='ds')
        # Plot dataset points
        fig.sp.plot_ds(dataset)
        # Plot objective function
        fig.switch_sptype(sp_type='function')
        fig.sp.plot_fobj(self.sgds[0].discriminant_function,
                         grid_limits=dataset.get_bounds())
        fig.sp.title('SGD Classifier')

        fig.show()

    def test_performance(self):
        """ Compare the classifiers performance"""
        self.logger.info("Testing error performance of the "
                         "classifiers on the training set")

        for sgd in self.sgds:

            self.logger.info("SGD kernel: {:}".format(sgd.kernel))

            svm = CClassifierSVM(sgd.kernel)

            svm.train(self.dataset)

            label_svm, y_svm = svm.classify(self.dataset.X)
            label_sgd, y_sgd = sgd.classify(self.dataset.X)

            acc_svm = CMetric.create('f1').performance_score(
                self.dataset.Y, label_svm)
            acc_sgd = CMetric.create('f1').performance_score(
                self.dataset.Y, label_sgd)

            self.logger.info("Accuracy of SVM: {:}".format(acc_svm))
            self.assertGreater(acc_svm, 0.90,
                               "Accuracy of SVM: {:}".format(acc_svm))
            self.logger.info("Accuracy of SGD: {:}".format(acc_sgd))
            self.assertGreater(acc_sgd, 0.90,
                               "Accuracy of SGD: {:}".format(acc_sgd))

    def test_margin(self):

        self.logger.info("Testing margin separation of SGD...")

        # we create 50 separable points
        dataset = CDLRandomBlobs(n_samples=50, centers=2, random_state=0,
                                 cluster_std=0.60).load()

        # fit the model
        clf = CClassifierSGD(loss=CLossHinge(), regularizer=CRegularizerL2(),
                             alpha=0.01, max_iter=200)
        clf.train(dataset)

        # plot the line, the points, and the nearest vectors to the plane
        xx = CArray.linspace(-1, 5, 10)
        yy = CArray.linspace(-1, 5, 10)

        X1, X2 = np.meshgrid(xx.tondarray(), yy.tondarray())
        Z = CArray.empty(X1.shape)
        for (i, j), val in np.ndenumerate(X1):
            x1 = val
            x2 = X2[i, j]
            Z[i, j] = clf.discriminant_function(CArray([x1, x2]))
        levels = [-1.0, 0.0, 1.0]
        linestyles = ['dashed', 'solid', 'dashed']
        colors = 'k'
        fig = CFigure(linewidth=1)
        fig.sp.contour(X1, X2, Z, levels, colors=colors, linestyles=linestyles)
        fig.sp.scatter(dataset.X[:, 0], dataset.X[:, 1], c=dataset.Y, s=40)

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

        for sgd in self.sgds:

            self.logger.info("SGD kernel: {:}".format(sgd.kernel))

            sgd.train(self.dataset)

            # Testing discriminant_function on multiple points

            df_scores_pos = sgd.discriminant_function(
                self.dataset.X, label=1)
            self.logger.info("discriminant_function("
                             "dataset.X, label=1:\n{:}".format(df_scores_pos))
            _check_df_scores(df_scores_pos, self.dataset.num_samples)

            df_scores_neg = sgd.discriminant_function(
                self.dataset.X, label=0)
            self.logger.info("discriminant_function("
                             "dataset.X, label=0:\n{:}".format(df_scores_neg))
            _check_df_scores(df_scores_neg, self.dataset.num_samples)

            self.assertFalse(
                ((df_scores_pos.sign() * -1) != df_scores_neg.sign()).any())

            # Testing _discriminant_function on multiple points

            ds_priv_scores = sgd._discriminant_function(
                self.dataset.X, label=1)
            self.logger.info("_discriminant_function("
                             "dataset.X, label=1:\n{:}".format(ds_priv_scores))
            _check_df_scores(ds_priv_scores, self.dataset.num_samples)

            # Comparing output of public and private

            self.assertFalse((df_scores_pos != ds_priv_scores).any())

            # Testing classify on multiple points

            labels, scores = sgd.classify(self.dataset.X)
            self.logger.info("classify(dataset.X:"
                             "\nlabels: {:}\nscores:{:}".format(labels,
                                                                scores))
            _check_classify_scores(
                labels, scores, self.dataset.num_samples, sgd.n_classes)

            # Comparing output of discriminant_function and classify

            self.assertFalse((df_scores_neg != scores[:, 0].ravel()).any())
            self.assertFalse((df_scores_pos != scores[:, 1].ravel()).any())

            # Testing discriminant_function on single point

            df_scores_pos = sgd.discriminant_function(
                self.dataset.X[0, :].ravel(), label=1)
            self.logger.info("discriminant_function(dataset.X[0, :].ravel(), "
                             "label=1:\n{:}".format(df_scores_pos))
            _check_df_scores(df_scores_pos, 1)

            df_scores_neg = sgd.discriminant_function(
                self.dataset.X[0, :].ravel(), label=0)
            self.logger.info("discriminant_function(dataset.X[0, :].ravel(), "
                             "label=0:\n{:}".format(df_scores_neg))
            _check_df_scores(df_scores_neg, 1)

            self.assertFalse(
                ((df_scores_pos.sign() * -1) != df_scores_neg.sign()).any())

            # Testing _discriminant_function on single point

            df_priv_scores = sgd._discriminant_function(
                self.dataset.X[0, :].ravel(), label=1)
            self.logger.info("_discriminant_function(dataset.X[0, :].ravel(), "
                             "label=1:\n{:}".format(df_priv_scores))
            _check_df_scores(df_priv_scores, 1)

            # Comparing output of public and private

            self.assertFalse((df_scores_pos != df_priv_scores).any())

            # Testing error raising

            with self.assertRaises(ValueError):
                sgd._discriminant_function(self.dataset.X, label=0)
            with self.assertRaises(ValueError):
                sgd._discriminant_function(
                    self.dataset.X[0, :].ravel(), label=0)

            self.logger.info("Testing classify on single point")

            labels, scores = sgd.classify(self.dataset.X[0, :].ravel())
            self.logger.info("classify(self.dataset.X[0, :].ravel():"
                             "\nlabels: {:}\nscores:{:}".format(labels,
                                                                scores))
            _check_classify_scores(labels, scores, 1, sgd.n_classes)

            # Comparing output of discriminant_function and classify

            self.assertFalse(
                (df_scores_neg != CArray(scores[:, 0]).ravel()).any())
            self.assertFalse(
                (df_scores_pos != CArray(scores[:, 1]).ravel()).any())


if __name__ == '__main__':
    CUnitTest.main()
