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

from secml.data.loader import CDataLoaderMNIST
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.ml.peval.metrics import CMetricAccuracy


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
        self.logger.info("Number of Patterns: %s",
                         str(self.dataset.num_samples))
        self.logger.info("Features: %s", str(self.dataset.num_features))

    def test_attributes(self):
        """Performs test on SVM attributes setting."""
        self.logger.info("Testing SVM attributes setting")

        for svm in self.svms:
            svm.set('C', 10)
            self.assertEqual(svm.C, 10)
            svm.set('class_weight', {-1: 1, 1: 50})
            # set gamma for poly and rbf and check if it is set properly
            if hasattr(svm.kernel, 'gamma'):
                svm.set('gamma', 100)
                self.assertEqual(svm.kernel.gamma, 100)

    def test_linear_svm(self):
        """Performs tests on linear SVM."""
        self.logger.info("Testing SVM linear variants (kernel and not)")

        # Instancing a linear SVM and an SVM with linear kernel
        linear_svm = CClassifierSVM(kernel=None)
        kernel_linear_svm = self.svms[0]

        self.logger.info("SVM w/ linear kernel in the primal")
        self.assertIsNone(linear_svm.kernel)

        self.logger.info("Training both classifiers on dense data")
        linear_svm.fit(self.dataset.X, self.dataset.Y)
        kernel_linear_svm.fit(self.dataset.X, self.dataset.Y)

        linear_svm_pred_y, linear_svm_pred_score = linear_svm.predict(
            self.dataset.X, return_decision_function=True)
        kernel_linear_svm_pred_y, \
        kernel_linear_svm_pred_score = kernel_linear_svm.predict(
            self.dataset.X, return_decision_function=True)

        # check prediction
        self.assert_array_equal(linear_svm_pred_y, kernel_linear_svm_pred_y)

        self.logger.info("Training both classifiers on sparse data")
        linear_svm.fit(self.dataset_sparse.X, self.dataset_sparse.Y)
        kernel_linear_svm.fit(self.dataset_sparse.X, self.dataset_sparse.Y)

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
                "SVM with kernel: %s", svm.kernel.__class__)

            # Training and predicting using our SVM
            svm.fit(self.dataset.X, self.dataset.Y)

            pred_y, pred_score = svm.predict(
                self.dataset.X, return_decision_function=True)

            # Training and predicting an SKlearn SVC
            k = svm.kernel.class_type if svm.kernel is not None else 'linear'
            sklearn_svm = SVC(kernel=k)

            # Setting similarity function parameters into SVC too
            # Exclude params not settable in sklearn_svm
            if svm.kernel is not None:
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
                             svm.kernel.__class__, accuracy)

    def test_shape(self):
        """Test shape of SVM parameters, scores etc."""
        import random

        self.logger.info("Testing SVM related vector shape")

        def _check_flattness(array):
            # self.assertEqual(len(array.shape) == 1, True)
            self.assertTrue(array.is_vector_like)

        for svm in self.svms:

            self.logger.info(
                "SVM with similarity function: %s", svm.kernel.__class__)

            # Training and predicting using our SVM
            svm.fit(self.dataset.X, self.dataset.Y)
            pred_y, pred_score = svm.predict(
                self.dataset.X, return_decision_function=True)
            # chose random one pattern
            pattern = CArray(random.choice(self.dataset.X.get_data()))
            gradient = svm.grad_f_x(pattern, y=1)

            if svm.w is not None:
                _check_flattness(svm.w)
            else:
                _check_flattness(svm.alpha)

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
                "SVM with similarity function: %s", svm.kernel.__class__)

            # Training and predicting on dense data for reference
            svm.fit(self.dataset.X, self.dataset.Y)
            pred_y, pred_score = svm.predict(
                self.dataset.X, return_decision_function=True)

            # Training and predicting on sparse data
            svm.fit(self.dataset_sparse.X, self.dataset_sparse.Y)
            pred_y_sparse, pred_score_sparse = svm.predict(
                self.dataset_sparse.X, return_decision_function=True)

            _check_sparsedata(pred_y, pred_score, pred_y_sparse,
                              pred_score_sparse)

            # Training on sparse and predicting on dense
            svm.fit(self.dataset_sparse.X, self.dataset_sparse.Y)
            pred_y_sparse, pred_score_sparse = svm.predict(
                self.dataset.X, return_decision_function=True)

            _check_sparsedata(pred_y, pred_score, pred_y_sparse,
                              pred_score_sparse)

            # Training on dense and predicting on sparse
            svm.fit(self.dataset.X, self.dataset.Y)
            pred_y_sparse, pred_score_sparse = svm.predict(
                self.dataset_sparse.X, return_decision_function=True)

            _check_sparsedata(pred_y, pred_score, pred_y_sparse,
                              pred_score_sparse)

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
        clf.fit(dataset.X, dataset.Y)

        w = clf.w
        a = -w[0] / w[1]

        xx = CArray.linspace(-5, 5)
        yy = a * xx - clf.b / w[1]

        wclf = CClassifierSVM(class_weight={0: 1, 1: 10})
        wclf.fit(dataset.X, dataset.Y)

        ww = wclf.w
        wa = -ww[0] / ww[1]
        wyy = wa * xx - wclf.b / ww[1]

        fig = CFigure(linewidth=1)
        fig.sp.plot(xx, yy.ravel(), 'k-', label='no weights')
        fig.sp.plot(xx, wyy.ravel(), 'k--', label='with weights')
        fig.sp.scatter(X[:, 0].ravel(), X[:, 1].ravel(), c=y)
        fig.sp.legend()

        fig.savefig(fm.join(fm.abspath(__file__), 'figs',
                            'test_c_classifier_svm.pdf'))

    def test_store_dual_vars(self):
        """Test of parameters that control storing of dual space variables."""
        self.logger.info("Checking CClassifierSVM.store_dual_vars...")

        self.logger.info("Linear SVM in primal space")
        svm = CClassifierSVM()
        svm.fit(self.dataset.X, self.dataset.Y)
        self.assertIsNone(svm.alpha)

        self.logger.info("Linear SVM in dual space")
        svm = CClassifierSVM(kernel='linear')
        svm.fit(self.dataset.X, self.dataset.Y)
        self.assertIsNotNone(svm.alpha)

        self.logger.info("Nonlinear SVM in dual space")
        svm = CClassifierSVM(kernel='rbf')
        svm.fit(self.dataset.X, self.dataset.Y)
        self.assertIsNotNone(svm.alpha)

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

            self.logger.info(
                "Computing gradient for SVM with kernel: %s", svm.kernel)

            if hasattr(svm.kernel, 'gamma'):  # set gamma for poly and rbf
                svm.set('gamma', 1e-5)
            if hasattr(svm.kernel, 'degree'):  # set degree for poly
                svm.set('degree', 3)

            samps = random.sample(range(self.dataset.num_samples), 5)

            self.logger.info("Testing dense data...")
            ds = self.dataset.todense()
            svm.fit(ds.X, ds.Y)

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
            svm.fit(ds.X, ds.Y)

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

    def test_multiclass(self):
        """Test multiclass SVM on MNIST digits."""

        self.logger.info("Testing multiclass SVM.")

        digits = tuple(range(0, 10))
        n_tr = 100  # Number of training set samples
        n_ts = 200  # Number of test set samples

        loader = CDataLoaderMNIST()
        tr = loader.load('training', digits=digits, num_samples=n_tr)
        ts = loader.load('testing', digits=digits, num_samples=n_ts)

        # Normalize the features in `[0, 1]`
        tr.X /= 255
        ts.X /= 255

        svm_params = {
            'kernel': CKernelRBF(gamma=0.1),
            'C': 10,
            'class_weight': {0: 1, 1: 1},
            'n_jobs': 2
        }
        classifiers = [
            CClassifierMulticlassOVA(CClassifierSVM, **svm_params),
            CClassifierSVM(**svm_params),
        ]

        grads = []
        acc = []
        for clf in classifiers:
            clf.verbose = 1
            # We can now fit the classifier
            clf.fit(tr.X, tr.Y)
            # Compute predictions on a test set
            y_pred, scores = clf.predict(ts.X, return_decision_function=True)
            # Evaluate the accuracy of the classifier
            metric = CMetricAccuracy()
            acc.append(metric.performance_score(y_true=ts.Y, y_pred=y_pred))
            grads.append(clf.grad_f_x(ts.X[1, :], 1))

        self.assertAlmostEqual(acc[0], acc[1])
        self.assert_array_almost_equal(grads[0], grads[1])


if __name__ == '__main__':
    CClassifierTestCases.main()
