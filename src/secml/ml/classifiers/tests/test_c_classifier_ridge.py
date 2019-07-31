from secml.ml.classifiers.tests import CClassifierTestCases

from secml.array import CArray
from secml.ml.classifiers import CClassifierRidge, CClassifierSVM
from secml.ml.kernel import *
from secml.data.loader import CDLRandom
from secml.ml.features.normalization import CNormalizerMinMax
from secml.ml.peval.metrics import CMetric
from secml.figure import CFigure


class TestCClassifierRidge(CClassifierTestCases):
    """Unit test for Ridge Classifier."""

    def setUp(self):
        """Test for init and fit methods."""
        # generate synthetic data
        self.dataset = CDLRandom(n_features=100, n_redundant=20,
                                 n_informative=25,
                                 n_clusters_per_class=2,
                                 random_state=0).load()

        self.dataset.X = CNormalizerMinMax().fit_transform(self.dataset.X)

        kernel_types = (None, CKernelLinear, CKernelRBF, CKernelPoly)
        self.ridges = [CClassifierRidge(
            kernel=kernel() if kernel is not None else None)
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
                            n_clusters_per_class=1, random_state=0).load()
        dataset.X = CNormalizerMinMax().fit_transform(dataset.X)

        self.ridges[0].fit(dataset)

        svm = CClassifierSVM()
        svm.fit(dataset)

        fig = CFigure(width=10, markersize=8)
        fig.subplot(2, 1, 1)
        # Plot dataset points
        fig.sp.plot_ds(dataset)
        # Plot objective function
        fig.sp.plot_fobj(svm.decision_function,
                         grid_limits=dataset.get_bounds(), y=1)
        fig.sp.title('SVM')

        fig.subplot(2, 1, 2)
        # Plot dataset points
        fig.sp.plot_ds(dataset)
        # Plot objective function
        fig.sp.plot_fobj(self.ridges[0].decision_function,
                         grid_limits=dataset.get_bounds(), y=1)
        fig.sp.title('ridge Classifier')

        fig.savefig('test_c_classifier_ridge.pdf')

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
        for ridge in self.ridges:
            self._test_fun(ridge, self.dataset.todense())
            self._test_fun(ridge, self.dataset.tosparse())

    def test_gradient(self):
        """Unittests for gradient_f_x."""
        self.logger.info("Testing Ridge.gradient_f_x() method")

        i = 5  # IDX of the point to test
        pattern = self.dataset.X[i, :]
        self.logger.info("P {:}: {:}".format(i, pattern))

        for ridge in self.ridges:

            self.logger.info(
                "Checking gradient for Ridge with kernel: %s", ridge.kernel)

            if hasattr(ridge.kernel, 'gamma'):  # set gamma for poly and rbf
                ridge.set('gamma', 1e-5)
            if hasattr(ridge.kernel, 'degree'):  # set degree for poly
                ridge.set('degree', 3)

            self.logger.info("Testing dense data...")
            ds = self.dataset.todense()
            ridge.fit(ds)

            # Run the comparison with numerical gradient
            # (all classes will be tested)
            grads_d = self._test_gradient_numerical(ridge, pattern.todense())

            self.logger.info("Testing sparse data...")
            ds = self.dataset.tosparse()
            ridge.fit(ds)

            # Run the comparison with numerical gradient
            # (all classes will be tested)
            grads_s = self._test_gradient_numerical(ridge, pattern.tosparse())

            # FIXME: WHY THIS TEST IS CRASHING? RANDOM_STATE MAYBE?
            # Compare dense gradients with sparse gradients
            # for grad_i, grad in enumerate(grads_d):
            #     self.assert_array_almost_equal(
            #         grad.atleast_2d(), grads_s[grad_i])

    def test_preprocess(self):
        """Test classifier with preprocessors inside."""
        ds = CDLRandom().load()
        clf = CClassifierRidge()

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
