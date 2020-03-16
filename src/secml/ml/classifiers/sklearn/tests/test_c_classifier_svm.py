from secml.ml.classifiers.tests import CClassifierTestCases

import numpy as np
from sklearn.svm import SVC
import sklearn.metrics as skm

from secml.data import CDataset
from secml.data.loader import CDLRandom
from secml.array import CArray
from secml.ml.classifiers import CClassifierSVM
from secml.figure import CFigure
from secml.ml.kernels import *
from secml.utils import fm


class TestCClassifierSVM(CClassifierTestCases):

    def setUp(self):

        # generate synthetic data
        self.dataset = CDLRandom(n_features=2, n_redundant=0, n_informative=1,
                                 n_clusters_per_class=1, random_state=1).load()

        self.dataset_sparse = self.dataset.tosparse()

        kernel_types = (None, CKernelLinear, CKernelRBF, CKernelPoly)
        self.svms = [CClassifierSVM(
            kernel=kernel() if kernel is not None else None)
                for kernel in kernel_types]
        self.logger.info(
            "Testing SVM with kernel functions: %s", str(kernel_types))

        for svm in self.svms:  # Enabling debug output for each classifier
            svm.verbose = 2

        self.logger.info("." * 50)
        self.logger.info("Number of Patterns: %s", str(self.dataset.num_samples))
        self.logger.info("Features: %s", str(self.dataset.num_features))

    def test_attributes(self):
        """Performs test on SVM attributes setting."""
        self.logger.info("Testing SVM attributes setting")

        for svm in self.svms:
            svm.set('C', 10)
            self.assertEqual(svm.C, 10)
            svm.set('class_weight', {-1: 1, 1: 50})
            # set gamma for poly and rbf in svm and check if it change also into the kernel
            if hasattr(svm.kernel, 'gamma'):
                svm.set('gamma', 100)
                self.assertEqual(svm.kernel.gamma, 100)

    def test_linear_svm(self):
        """Performs tests on linear SVM."""
        self.logger.info("Testing SVM linear variants (kernel and not)")

        # Instancing a linear SVM and an SVM with linear kernel
        linear_svm = CClassifierSVM(kernel=None)
        kernel_linear_svm = self.svms[0]

        self.logger.info("SVM kernel: {:}".format(linear_svm.kernel))
        self.assertEqual('linear', linear_svm.kernel.class_type)

        self.logger.info("Training both classifiers on dense data")
        linear_svm.fit(self.dataset)
        kernel_linear_svm.fit(self.dataset)

        linear_svm_pred_y, linear_svm_pred_score = linear_svm.predict(
            self.dataset.X, return_decision_function=True)
        kernel_linear_svm_pred_y, \
            kernel_linear_svm_pred_score = kernel_linear_svm.predict(
                self.dataset.X, return_decision_function=True)

        # check prediction
        self.assert_array_equal(linear_svm_pred_y, kernel_linear_svm_pred_y)

        self.logger.info("Training both classifiers on sparse data")
        linear_svm.fit(self.dataset_sparse)
        kernel_linear_svm.fit(self.dataset_sparse)

        self.assertTrue(linear_svm.w.issparse,
                        "Weights vector is not sparse even "
                        "if training data is sparse")

        linear_svm_pred_y, linear_svm_pred_score = linear_svm.predict(
            self.dataset_sparse.X, return_decision_function=True)
        kernel_linear_svm_pred_y, \
            kernel_linear_svm_pred_score = kernel_linear_svm.predict(
                self.dataset_sparse.X, return_decision_function=True)

        # check prediction
        self.assert_array_equal(linear_svm_pred_y, kernel_linear_svm_pred_y)

    def test_predict(self):
        """Performs tests on SVM prediction capabilities."""
        self.logger.info("Testing SVM predict accuracy")

        for svm in self.svms:
            self.logger.info(
                "SVM \\w similarity function: %s", svm.kernel.__class__)

            # Training and predicting using our SVM
            svm.fit(self.dataset)

            pred_y, pred_score = svm.predict(
                self.dataset.X, return_decision_function=True)

            # Training and predicting an SKlearn SVC
            sklearn_svm = SVC(kernel=svm.kernel.class_type)

            # Setting similarity function parameters into SVC too
            # Exclude params not settable in sklearn_svm
            p_dict = {}
            for p in svm.kernel.get_params():
                if p in sklearn_svm.get_params():
                    p_dict[p] = svm.kernel.get_params()[p]
            sklearn_svm.set_params(**p_dict)

            sklearn_svm.fit(self.dataset.X.get_data(),
                            np.ravel(self.dataset.Y.get_data()))
            sklearn_pred_y = sklearn_svm.predict(self.dataset.X.get_data())
            sklearn_score = sklearn_svm.decision_function(
                self.dataset.X.get_data())

            # Test if sklearn pred_y are equal to our predicted labels
            self.assert_array_equal(pred_y, sklearn_pred_y)
            # Test if sklearn computed distance from separating hyperplane
            # is the same of own. This is a fix for some architectures that
            # exhibit floating point problems
            self.assert_allclose(pred_score[:, 1].ravel(), sklearn_score)

            # EVALUATE PERFORMANCE
            accuracy = skm.accuracy_score(
                self.dataset.Y.get_data(), sklearn_pred_y)
            self.logger.info("Prediction accuracy for kernel %s is %f ",
                             svm.kernel.class_type, accuracy)

    def test_shape(self):
        """Test shape of SVM parameters, scores etc."""
        import random

        self.logger.info("Testing SVM related vector shape")

        def _check_flattness(array):
            self.assertEqual(len(array.shape) == 1, True)

        for svm in self.svms:

            self.logger.info(
                "SVM \\w similarity function: %s", svm.kernel.__class__)

            # Training and predicting using our SVM
            svm.fit(self.dataset)
            pred_y, pred_score = svm.predict(
                self.dataset.X, return_decision_function=True)
            # chose random one pattern
            pattern = CArray(random.choice(self.dataset.X.get_data()))
            gradient = svm.grad_f_x(pattern)

            if svm.is_kernel_linear():
                _check_flattness(svm.w)
            else:
                _check_flattness(svm.alpha)
                _check_flattness(svm.sv_idx)

            _check_flattness(pred_y)
            _check_flattness(gradient)


    def test_sparse(self):
        """Performs tests on sparse dataset."""
        self.logger.info("Testing SVM on sparse data")

        def _check_sparsedata(y, score, y_sparse, score_sparse):

            self.assertFalse((y != y_sparse).any(),
                             "Predicted labels on sparse data are different.")
            # Rounding scores to prevent false positives in assert
            score_rounded = score[:, 1].ravel().round(3)
            score_sparse_rounded = score_sparse[:, 1].ravel().round(3)
            self.assertFalse((score_rounded != score_sparse_rounded).any(),
                             "Predicted Scores on sparse data are different.")

        for svm in self.svms:
            self.logger.info(
                "SVM \\w similarity function: %s", svm.kernel.__class__)

            # Training and predicting on dense data for reference
            svm.fit(self.dataset)
            pred_y, pred_score = svm.predict(
                self.dataset.X, return_decision_function=True)

            # Training and predicting on sparse data
            svm.fit(self.dataset_sparse)
            pred_y_sparse, pred_score_sparse = svm.predict(
                self.dataset_sparse.X, return_decision_function=True)

            _check_sparsedata(pred_y, pred_score, pred_y_sparse, pred_score_sparse)

            # Training on sparse and predicting on dense
            svm.fit(self.dataset_sparse)
            pred_y_sparse, pred_score_sparse = svm.predict(
                self.dataset.X, return_decision_function=True)

            _check_sparsedata(pred_y, pred_score, pred_y_sparse, pred_score_sparse)

            # Training on dense and predicting on sparse
            svm.fit(self.dataset)
            pred_y_sparse, pred_score_sparse = svm.predict(
                self.dataset_sparse.X, return_decision_function=True)

            _check_sparsedata(pred_y, pred_score, pred_y_sparse, pred_score_sparse)

    def test_margin(self):
        self.logger.info("Testing margin separation of SVM...")

        import numpy as np

        # we create 40 separable points
        rng = np.random.RandomState(0)
        n_samples_1 = 1000
        n_samples_2 = 100
        X = np.r_[1.5 * rng.randn(n_samples_1, 2),
                  0.5 * rng.randn(n_samples_2, 2) + [2, 2]]
        y = [0] * (n_samples_1) + [1] * (n_samples_2)

        dataset = CDataset(X, y)

        # fit the model
        clf = CClassifierSVM()
        clf.fit(dataset)

        w = clf.w
        a = -w[0] / w[1]

        xx = CArray.linspace(-5, 5)
        yy = a * xx - clf.b / w[1]

        wclf = CClassifierSVM(class_weight={0: 1, 1: 10})
        wclf.fit(dataset)

        ww = wclf.w
        wa = -ww[0] / ww[1]
        wyy = wa * xx - wclf.b / ww[1]

        fig = CFigure(linewidth=1)
        fig.sp.plot(xx, yy, 'k-', label='no weights')
        fig.sp.plot(xx, wyy, 'k--', label='with weights')
        fig.sp.scatter(X[:, 0].ravel(), X[:, 1].ravel(), c=y)
        fig.sp.legend()

        fig.savefig(fm.join(fm.abspath(__file__), 'figs',
                            'test_c_classifier_svm.pdf'))

    def test_store_dual_vars(self):
        """Test of parameters that control storing of dual space variables."""
        self.logger.info("Checking CClassifierSVM.store_dual_vars...")

        self.logger.info("Instancing a linear SVM")
        svm = CClassifierSVM(kernel=None)

        self.assertIsNone(svm.store_dual_vars)
        svm.fit(self.dataset)
        self.assertIsNone(svm.sv)

        self.logger.info("Changing store_dual_vars to True")
        svm.store_dual_vars = True

        self.assertTrue(svm.store_dual_vars)
        svm.fit(self.dataset)
        self.assertIsNotNone(svm.sv)

        self.logger.info("Changing store_dual_vars to False")
        svm.store_dual_vars = False

        self.assertFalse(svm.store_dual_vars)
        svm.fit(self.dataset)
        self.assertIsNone(svm.sv)

        self.logger.info("Changing kernel to nonlinear when "
                         "store_dual_vars is False should raise ValueError")
        with self.assertRaises(ValueError):
            svm.kernel = CKernelRBF()

        self.logger.info("Instancing a nonlinear SVM")
        svm = CClassifierSVM(kernel='rbf')

        self.assertIsNone(svm.store_dual_vars)
        svm.fit(self.dataset)
        self.assertIsNotNone(svm.sv)

        self.logger.info("Changing store_dual_vars to True")
        svm.store_dual_vars = True

        self.assertTrue(svm.store_dual_vars)
        svm.fit(self.dataset)
        self.assertIsNotNone(svm.sv)

        self.logger.info(
            "Changing store_dual_vars to False should raise ValueError")
        with self.assertRaises(ValueError):
            svm.store_dual_vars = False

    def test_fun(self):
        """Test for decision_function() and predict() methods."""
        for clf in self.svms:

            self.logger.info("SVM kernel: {:}".format(clf.kernel))

            scores_d = self._test_fun(clf, self.dataset.todense())
            scores_s = self._test_fun(clf, self.dataset.tosparse())

            self.assert_array_almost_equal(scores_d, scores_s)

    def test_gradient(self):
        """Performs tests on gradient."""
        self.logger.info("Testing SVM.gradient() method")

        import random
        for svm in self.svms:

            self.logger.info("Computing gradient for SVM with kernel: %s",
                             svm.kernel.class_type)

            if hasattr(svm.kernel, 'gamma'):  # set gamma for poly and rbf
                svm.set('gamma', 1e-5)
            if hasattr(svm.kernel, 'degree'):  # set degree for poly
                svm.set('degree', 3)

            samps = random.sample(range(self.dataset.num_samples), 5)

            self.logger.info("Testing dense data...")
            ds = self.dataset.todense()
            svm.fit(ds)

            grads_d = []
            for i in samps:
                # Randomly extract a pattern to test
                pattern = ds.X[i, :]
                self.logger.info("P {:}: {:}".format(i, pattern))
                # Run the comparison with numerical gradient
                # (all classes will be tested)
                grads_d += self._test_gradient_numerical(svm, pattern)

            self.logger.info("Testing sparse data...")
            ds = self.dataset.tosparse()
            svm.fit(ds)

            grads_s = []
            for i in samps:
                # Randomly extract a pattern to test
                pattern = ds.X[i, :]
                self.logger.info("P {:}: {:}".format(i, pattern))
                # Run the comparison with numerical gradient
                # (all classes will be tested)
                grads_s += self._test_gradient_numerical(svm, pattern)

            # Compare dense gradients with sparse gradients
            for grad_i, grad in enumerate(grads_d):
                self.assert_array_almost_equal(
                    grad.atleast_2d(), grads_s[grad_i])

    def test_preprocess(self):
        """Test classifier with preprocessors inside."""
        ds = CDLRandom().load()
        clf = CClassifierSVM()

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
