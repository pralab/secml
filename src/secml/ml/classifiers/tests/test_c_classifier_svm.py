from . import CClassifierTestCases

import numpy as np
from sklearn.svm import SVC
import sklearn.metrics as skm

from secml.data import CDataset
from secml.data.loader import CDLRandom
from secml.array import CArray
from secml.ml.classifiers import CClassifierSVM
from secml.figure import CFigure
from secml.ml.kernel import *


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
        self.logger.info("Testing SVM linear variants (kernel and linear model)")

        # Instancing a linear SVM and an SVM with linear kernel
        linear_svm = CClassifierSVM(kernel=None)
        kernel_linear_svm = self.svms[0]

        self.logger.info("SVM kernel: {:}".format(linear_svm.kernel))
        self.assertEquals(linear_svm.kernel.class_type, 'linear')

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
                "SVM \w similarity function: %s", svm.kernel.__class__)

            # Training and predicting using our SVM
            svm.fit(self.dataset)

            pred_y, pred_score = svm.predict(
                self.dataset.X, return_decision_function=True)

            # Training and predicting an SKlearn SVC
            sklearn_svm = SVC(kernel=svm.kernel.class_type)

            # Setting similarity function parameters into SVC too
            sklearn_svm.set_params(**svm.kernel.get_params())

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
            self.assertEqual(len(array.shape) == 1, True, "shape is not correct")

        for svm in self.svms:

            self.logger.info("SVM \w similarity function: %s", svm.kernel.__class__)

            # Training and predicting using our SVM
            svm.fit(self.dataset)
            pred_y, pred_score = svm.predict(
                self.dataset.X, return_decision_function=True)
            # chose random one pattern
            pattern = CArray(random.choice(self.dataset.X.get_data()))
            gradient = svm.gradient_f_x(pattern)

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

            self.assertFalse((y != y_sparse).any(), "Predicted labels on sparse data are different.")
            # Rounding scores to prevent false positives in assert
            score_rounded = score[:, 1].ravel().round(3)
            score_sparse_rounded = score_sparse[:, 1].ravel().round(3)
            self.assertFalse((score_rounded != score_sparse_rounded).any(),
                             "Predicted Scores on sparse data are different.")

        for svm in self.svms:
            self.logger.info("SVM \w similarity function: %s", svm.kernel.__class__)

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

            svm.fit(self.dataset)

            for i in random.sample(xrange(self.dataset.num_samples), 10):
                # Randomly extract a pattern to test
                pattern = self.dataset.X[i, :]
                self.logger.info("P {:}: {:}".format(i, pattern))
                # Run the comparison with numerical gradient
                # (all classes will be tested)
                self._test_gradient_numerical(svm, pattern)

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

        fig.show()

    def test_normalizer(self):

        from secml.ml.features.normalization import CNormalizerMinMax

        data = CDLRandom().load()
        norm = CNormalizerMinMax()
        data_norm = norm.fit_normalize(data.X)

        svm1 = CClassifierSVM(preprocess='min-max')
        svm2 = CClassifierSVM()

        svm1.fit(data)
        y1, score1 = svm1.predict(data.X, return_decision_function=True)

        svm2.fit(CDataset(data_norm, data.Y))
        y2, score2 = svm2.predict(data_norm, return_decision_function=True)

        self.assertTrue((y1 == y2).all())
        self.assertTrue((score1[:, 0] == score2[:, 0]).all())

        svm1_grad = svm1.gradient_f_x(data.X[0, :])
        svm2_grad = svm2.gradient_f_x(data_norm[0, :]) * norm.gradient(
            data_norm[0, :]).diag()

        self.assertTrue((svm1_grad == svm2_grad).all())

    def test_store_dual_vars(self):
        """Test of parameters that control storing of dual space variables."""
        self.logger.info("Checking CClassifierSVM.store_dual_vars...")

        self.logger.info("Instancing a linear SVM")
        svm = CClassifierSVM(kernel=None)

        self.assertEquals(svm.store_dual_vars, None)
        svm.fit(self.dataset)
        self.assertEquals(svm.sv, None)

        self.logger.info("Changing store_dual_vars to True")
        svm.store_dual_vars = True

        self.assertEquals(svm.store_dual_vars, True)
        svm.fit(self.dataset)
        self.assertNotEquals(svm.sv, None)

        self.logger.info("Changing store_dual_vars to False")
        svm.store_dual_vars = False

        self.assertEquals(svm.store_dual_vars, False)
        svm.fit(self.dataset)
        self.assertEquals(svm.sv, None)

        self.logger.info("Changing kernel to nonlinear when "
                         "store_dual_vars is False should raise ValueError")
        with self.assertRaises(ValueError):
            svm.kernel = CKernelRBF()

        self.logger.info("Instancing a nonlinear SVM")
        svm = CClassifierSVM(kernel='rbf')

        self.assertEquals(svm.store_dual_vars, None)
        svm.fit(self.dataset)
        self.assertNotEquals(svm.sv, None)

        self.logger.info("Changing store_dual_vars to True")
        svm.store_dual_vars = True

        self.assertEquals(svm.store_dual_vars, True)
        svm.fit(self.dataset)
        self.assertNotEquals(svm.sv, None)

        self.logger.info(
            "Changing store_dual_vars to False should raise ValueError")
        with self.assertRaises(ValueError):
            svm.store_dual_vars = False

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

        for svm in self.svms:

            self.logger.info("SVM kernel: {:}".format(svm.kernel))

            svm.fit(self.dataset)

            x = x_norm = self.dataset.X
            p = p_norm = self.dataset.X[0, :].ravel()

            # Preprocessing data if a preprocess is defined
            if svm.preprocess is not None:
                x_norm = svm.preprocess.normalize(x)
                p_norm = svm.preprocess.normalize(p)

            # Testing decision_function on multiple points

            df_scores_neg = svm.decision_function(x, y=0)
            self.logger.info("decision_function(x, y=0):\n"
                             "{:}".format(df_scores_neg))
            _check_df_scores(df_scores_neg, self.dataset.num_samples)

            df_scores_pos = svm.decision_function(x, y=1)
            self.logger.info("decision_function(x, y=1):\n"
                             "{:}".format(df_scores_pos))
            _check_df_scores(df_scores_pos, self.dataset.num_samples)

            self.assertFalse(
                ((df_scores_pos.sign() * -1) != df_scores_neg.sign()).any())

            # Testing _decision_function on multiple points

            ds_priv_scores = svm._decision_function(x_norm, y=1)
            self.logger.info("_decision_function(x_norm, y=1):\n"
                             "{:}".format(ds_priv_scores))
            _check_df_scores(ds_priv_scores, self.dataset.num_samples)

            # Comparing output of public and private

            self.assertFalse((df_scores_pos != ds_priv_scores).any())

            # Testing predict on multiple points

            labels, scores = svm.predict(
                self.dataset.X, return_decision_function=True)
            self.logger.info("predict(x):\nlabels: {:}\n"
                             "scores: {:}".format(labels, scores))
            _check_classify_scores(
                labels, scores, self.dataset.num_samples, svm.n_classes)

            # Comparing output of decision_function and predict

            self.assertFalse((df_scores_neg != scores[:, 0].ravel()).any())
            self.assertFalse((df_scores_pos != scores[:, 1].ravel()).any())

            # Testing decision_function on single point

            df_scores_neg = svm.decision_function(p, y=0)
            self.logger.info("decision_function(p, y=0):\n"
                             "{:}".format(df_scores_neg))
            _check_df_scores(df_scores_neg, 1)

            df_scores_pos = svm.decision_function(p, y=1)
            self.logger.info("decision_function(p, y=1):\n"
                             "{:}".format(df_scores_pos))
            _check_df_scores(df_scores_pos, 1)

            self.assertFalse(
                ((df_scores_pos.sign() * -1) != df_scores_neg.sign()).any())

            # Testing _decision_function on single point

            df_priv_scores = svm._decision_function(p_norm, y=1)
            self.logger.info("_decision_function(p_norm, y=1):\n"
                             "{:}".format(df_priv_scores))
            _check_df_scores(df_priv_scores, 1)

            # Comparing output of public and private

            self.assertFalse((df_scores_pos != df_priv_scores).any())

            self.logger.info("Testing predict on single point")

            labels, scores = svm.predict(p, return_decision_function=True)
            self.logger.info("predict(p):\nlabels: {:}\n"
                             "scores: {:}".format(labels, scores))
            _check_classify_scores(labels, scores, 1, svm.n_classes)

            # Comparing output of decision_function and predict

            self.assertFalse(
                (df_scores_neg != CArray(scores[:, 0]).ravel()).any())
            self.assertFalse(
                (df_scores_pos != CArray(scores[:, 1]).ravel()).any())

            # Testing error raising

            with self.assertRaises(ValueError):
                svm._decision_function(x_norm, y=0)
            with self.assertRaises(ValueError):
                svm._decision_function(p_norm, y=0)


if __name__ == '__main__':
    CClassifierTestCases.main()
