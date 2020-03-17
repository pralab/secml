from secml.ml.classifiers.tests import CClassifierTestCases

import numpy as np

from secml.ml.classifiers import CClassifierSGD, CClassifierSVM
from secml.ml.classifiers.regularizer import *
from secml.ml.classifiers.loss import *
from secml.ml.kernels import *
from secml.array import CArray
from secml.data.loader import CDLRandom, CDLRandomBlobs
from secml.ml.features.normalization import CNormalizerMinMax
from secml.ml.peval.metrics import CMetric
from secml.figure import CFigure
from secml.utils import fm


class TestCClassifierSGD(CClassifierTestCases):
    """Unit test for SGD Classifier."""

    def setUp(self):
        """Test for init and fit methods."""

        # TODO: remove this filter when `kernel` parameter is removed from SGD Classifier
        self.logger.filterwarnings("ignore", message="`kernel` parameter.*",
                                   category=DeprecationWarning)

        # generate synthetic data
        self.dataset = CDLRandom(n_features=100, n_redundant=20,
                                 n_informative=25,
                                 n_clusters_per_class=2,
                                 random_state=0).load()

        self.dataset.X = CNormalizerMinMax().fit_transform(self.dataset.X)

        self.logger.info("Testing classifier creation ")
        self.sgd = CClassifierSGD(regularizer=CRegularizerL2(),
                                  loss=CLossHinge(),
                                  random_state=0)

        kernel_types = \
            (None, CKernelLinear(), CKernelRBF(), CKernelPoly(degree=3))
        self.sgds = [CClassifierSGD(
            regularizer=CRegularizerL2(), loss=CLossHinge(),
            max_iter=500, random_state=0,
            kernel=kernel if kernel is not None else None)
                for kernel in kernel_types]
        self.logger.info(
            "Testing SGD with kernel functions: %s", str(kernel_types))

        for sgd in self.sgds:
            sgd.verbose = 2  # Enabling debug output for each classifier
            sgd.fit(self.dataset)

    def test_draw(self):
        """ Compare the classifiers graphically"""
        self.logger.info("Testing classifiers graphically")

        # generate 2D synthetic data
        dataset = CDLRandom(n_features=2, n_redundant=1, n_informative=1,
                            n_clusters_per_class=1).load()
        dataset.X = CNormalizerMinMax().fit_transform(dataset.X)

        self.sgds[0].fit(dataset)

        svm = CClassifierSVM()
        svm.fit(dataset)

        fig = CFigure(width=10, markersize=8)
        fig.subplot(2, 1, 1)
        # Plot dataset points
        fig.sp.plot_ds(dataset)
        # Plot objective function
        fig.sp.plot_fun(svm.decision_function,
                        grid_limits=dataset.get_bounds(), y=1)
        fig.sp.title('SVM')

        fig.subplot(2, 1, 2)
        # Plot dataset points
        fig.sp.plot_ds(dataset)
        # Plot objective function
        fig.sp.plot_fun(self.sgds[0].decision_function,
                        grid_limits=dataset.get_bounds(), y=1)
        fig.sp.title('SGD Classifier')

        fig.savefig(fm.join(fm.abspath(__file__), 'figs',
                            'test_c_classifier_sgd1.pdf'))

    def test_performance(self):
        """ Compare the classifiers performance"""
        self.logger.info("Testing error performance of the "
                         "classifiers on the training set")

        for sgd in self.sgds:

            self.logger.info("SGD kernel: {:}".format(sgd.kernel))

            svm = CClassifierSVM(sgd.kernel)

            svm.fit(self.dataset)

            label_svm, y_svm = svm.predict(
                self.dataset.X, return_decision_function=True)
            label_sgd, y_sgd = sgd.predict(
                self.dataset.X, return_decision_function=True)

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
                             alpha=0.01, max_iter=200, random_state=0)
        clf.fit(dataset)

        # plot the line, the points, and the nearest vectors to the plane
        xx = CArray.linspace(-1, 5, 10)
        yy = CArray.linspace(-1, 5, 10)

        X1, X2 = np.meshgrid(xx.tondarray(), yy.tondarray())
        Z = CArray.empty(X1.shape)
        for (i, j), val in np.ndenumerate(X1):
            x1 = val
            x2 = X2[i, j]
            Z[i, j] = clf.decision_function(CArray([x1, x2]), y=1)
        levels = [-1.0, 0.0, 1.0]
        linestyles = ['dashed', 'solid', 'dashed']
        colors = 'k'
        fig = CFigure(linewidth=1)
        fig.sp.contour(X1, X2, Z, levels, colors=colors, linestyles=linestyles)
        fig.sp.scatter(dataset.X[:, 0].ravel(),
                       dataset.X[:, 1].ravel(),
                       c=dataset.Y, s=40)

        fig.savefig(fm.join(fm.abspath(__file__), 'figs',
                            'test_c_classifier_sgd2.pdf'))

    def test_fun(self):
        """Test for decision_function() and predict() methods."""
        for clf in self.sgds:

            self.logger.info("SGD kernel: {:}".format(clf.kernel))

            scores_d = self._test_fun(clf, self.dataset.todense())
            scores_s = self._test_fun(clf, self.dataset.tosparse())

            # FIXME: WHY THIS TEST IS CRASHING? RANDOM_STATE MAYBE?
            # self.assert_array_almost_equal(scores_d, scores_s)

    def test_gradient(self):
        """Unittests for gradient_f_x."""
        self.logger.info("Testing SGD.gradient_f_x() method")

        i = 5  # IDX of the point to test
        pattern = self.dataset.X[i, :]
        self.logger.info("P {:}: {:}".format(i, pattern))

        for sgd in self.sgds:

            self.logger.info(
                "Checking gradient for SGD with kernel: %s", sgd.kernel)

            if hasattr(sgd.kernel, 'gamma'):  # set gamma for poly and rbf
                sgd.set('gamma', 1e-5)
            if hasattr(sgd.kernel, 'degree'):  # set degree for poly
                sgd.set('degree', 3)

            self.logger.info("Testing dense data...")
            ds = self.dataset.todense()
            sgd.fit(ds)

            # Run the comparison with numerical gradient
            # (all classes will be tested)
            grads_d = self._test_gradient_numerical(sgd, pattern.todense())

            self.logger.info("Testing sparse data...")
            ds = self.dataset.tosparse()
            sgd.fit(ds)

            # Run the comparison with numerical gradient
            # (all classes will be tested)
            grads_s = self._test_gradient_numerical(sgd, pattern.tosparse())

            # FIXME: WHY THIS TEST IS CRASHING? RANDOM_STATE MAYBE?
            # Compare dense gradients with sparse gradients
            # for grad_i, grad in enumerate(grads_d):
            #     self.assert_array_almost_equal(
            #         grad.atleast_2d(), grads_s[grad_i])

    def test_preprocess(self):
        """Test classifier with preprocessors inside."""
        ds = CDLRandom().load()
        clf = CClassifierSGD(
            regularizer=CRegularizerL2(), loss=CLossHinge(), random_state=0)

        # All linear transformations with gradient implemented
        self._test_preprocess(ds, clf,
                              ['min-max', 'mean-std'],
                              [{'feature_range': (-1, 1)}, {}])
        self._test_preprocess_grad(ds, clf,
                                   ['min-max', 'mean-std'],
                                   [{'feature_range': (-1, 1)}, {}])

        # Mixed linear/nonlinear transformations without gradient
        self._test_preprocess(ds, clf, ['pca', 'unit-norm'], [{}, {}])


if __name__ == '__main__':
    CClassifierTestCases.main()
