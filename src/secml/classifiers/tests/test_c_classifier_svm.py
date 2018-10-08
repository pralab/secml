from secml.utils import CUnitTest

import numpy as np
import numpy.testing as npt
from sklearn.svm import SVC
import sklearn.metrics as skm

from secml.data import CDataset
from secml.data.loader import CDLRandom
from secml.array import CArray
from secml.classifiers import CClassifierSVM
from secml.figure import CFigure
from secml.kernel import *
from secml.optimization import COptimizer
from secml.optimization.function import CFunction


class TestCClassifierSVM(CUnitTest):

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
        linear_svm.train(self.dataset)
        kernel_linear_svm.train(self.dataset)

        linear_svm_pred_y, linear_svm_pred_score = linear_svm.classify(self.dataset.X)
        kernel_linear_svm_pred_y, kernel_linear_svm_pred_score = kernel_linear_svm.classify(self.dataset.X)

        # check prediction
        npt.assert_array_equal(linear_svm_pred_y.get_data(), kernel_linear_svm_pred_y.get_data())

        self.logger.info("Training both classifiers on sparse data")
        linear_svm.train(self.dataset_sparse)
        kernel_linear_svm.train(self.dataset_sparse)

        self.assertTrue(linear_svm.w.issparse, "Weights vector is not sparse even if training data is sparse")

        linear_svm_pred_y, linear_svm_pred_score = linear_svm.classify(self.dataset_sparse.X)
        kernel_linear_svm_pred_y, kernel_linear_svm_pred_score = kernel_linear_svm.classify(self.dataset_sparse.X)

        # check prediction
        npt.assert_array_equal(linear_svm_pred_y.get_data(), kernel_linear_svm_pred_y.get_data())

    def test_predict(self):
        """Performs tests on SVM prediction capabilities."""
        self.logger.info("Testing SVM predict accuracy")

        for svm in self.svms:
            self.logger.info("SVM \w similarity function: %s", svm.kernel.__class__)

            # Training and predicting using our SVM
            svm.train(self.dataset)

            pred_y, pred_score = svm.classify(self.dataset.X)

            # Training and predicting an SKlearn SVC
            sklearn_svm = SVC(kernel=svm.kernel.class_type)

            # Setting similarity function parameters into SVC too
            sklearn_svm.set_params(**svm.kernel.get_params())

            sklearn_svm.fit(self.dataset.X.get_data(), np.ravel(self.dataset.Y.get_data()))
            sklearn_pred_y = sklearn_svm.predict(self.dataset.X.get_data())
            sklearn_score = sklearn_svm.decision_function(self.dataset.X.get_data())

            # Test if sklearn predicted labels are equal to our predicted labels
            npt.assert_array_equal(pred_y.get_data(), sklearn_pred_y)
            # Test if sklearn computed distance from separating hyperplane is the same of own
            # This is a fix for some architectures that exhibit floating point problems
            npt.assert_allclose(pred_score[:, 1].get_data().ravel(), sklearn_score)

            # EVALUATE PERFORMANCE
            accuracy = skm.accuracy_score(self.dataset.Y.get_data(), sklearn_pred_y)
            self.logger.info("Prediction accuracy for kernel %s is %f ", svm.kernel.class_type, accuracy)

    def test_shape(self):
        """Test shape of SVM parameters, scores etc."""
        import random

        self.logger.info("Testing SVM related vector shape")

        def _check_flattness(array):
            self.assertEqual(len(array.shape) == 1, True, "shape is not correct")

        for svm in self.svms:

            self.logger.info("SVM \w similarity function: %s", svm.kernel.__class__)

            # Training and predicting using our SVM
            svm.train(self.dataset)
            pred_y, pred_score = svm.classify(self.dataset.X)
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
            svm.train(self.dataset)
            pred_y, pred_score = svm.classify(self.dataset.X)

            # Training and predicting on sparse data
            svm.train(self.dataset_sparse)
            pred_y_sparse, pred_score_sparse = svm.classify(self.dataset_sparse.X)

            _check_sparsedata(pred_y, pred_score, pred_y_sparse, pred_score_sparse)

            # Training on sparse and predicting on dense
            svm.train(self.dataset_sparse)
            pred_y_sparse, pred_score_sparse = svm.classify(self.dataset.X)

            _check_sparsedata(pred_y, pred_score, pred_y_sparse, pred_score_sparse)

            # Training on dense and predicting on sparse
            svm.train(self.dataset)
            pred_y_sparse, pred_score_sparse = svm.classify(self.dataset_sparse.X)

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

            svm.train(self.dataset)

            for i in random.sample(xrange(self.dataset.num_samples), 10):
                pattern = self.dataset.X[i, :]

                self.logger.info("P {:}: {:}".format(i, pattern))

                # Compare the analytical grad with the numerical grad
                gradient = svm.gradient_f_x(pattern, y=1)
                self.logger.info("Gradient: %s", str(gradient))
                check_grad_val = COptimizer(
                    CFunction(svm.discriminant_function,
                              svm._gradient_f)).check_grad(pattern)
                self.logger.info(
                    "norm(grad - num_grad): %s", str(check_grad_val))
                self.assertLess(check_grad_val, 1e-3,
                                "problematic kernel is " +
                                svm.kernel.class_type)
                for i, elm in enumerate(gradient):
                    self.assertIsInstance(elm, float)

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
        clf.train(dataset)

        w = clf.w
        a = -w[0] / w[1]

        xx = CArray.linspace(-5, 5)
        yy = a * xx - clf.b / w[1]

        wclf = CClassifierSVM(class_weight={0: 1, 1: 10})
        wclf.train(dataset)

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

        from secml.features.normalization import CNormalizerMinMax

        data = CDLRandom().load()
        norm = CNormalizerMinMax()
        data_norm = norm.train_normalize(data.X)

        svm1 = CClassifierSVM(normalizer='minmax')
        svm2 = CClassifierSVM()

        svm1.train(data)
        y1, score1 = svm1.classify(data.X)

        svm2.train(CDataset(data_norm, data.Y))
        y2, score2 = svm2.classify(data_norm)

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
        svm.train(self.dataset)
        self.assertEquals(svm.sv, None)

        self.logger.info("Changing store_dual_vars to True")
        svm.store_dual_vars = True

        self.assertEquals(svm.store_dual_vars, True)
        svm.train(self.dataset)
        self.assertNotEquals(svm.sv, None)

        self.logger.info("Changing store_dual_vars to False")
        svm.store_dual_vars = False

        self.assertEquals(svm.store_dual_vars, False)
        svm.train(self.dataset)
        self.assertEquals(svm.sv, None)

        self.logger.info("Changing kernel to nonlinear when "
                         "store_dual_vars is False should raise ValueError")
        with self.assertRaises(ValueError):
            svm.kernel = CKernelRBF()

        self.logger.info("Instancing a nonlinear SVM")
        svm = CClassifierSVM(kernel='rbf')

        self.assertEquals(svm.store_dual_vars, None)
        svm.train(self.dataset)
        self.assertNotEquals(svm.sv, None)

        self.logger.info("Changing store_dual_vars to True")
        svm.store_dual_vars = True

        self.assertEquals(svm.store_dual_vars, True)
        svm.train(self.dataset)
        self.assertNotEquals(svm.sv, None)

        self.logger.info(
            "Changing store_dual_vars to False should raise ValueError")
        with self.assertRaises(ValueError):
            svm.store_dual_vars = False

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

        for svm in self.svms:

            self.logger.info("SVM kernel: {:}".format(svm.kernel))

            svm.train(self.dataset)

            x = x_norm = self.dataset.X
            p = p_norm = self.dataset.X[0, :].ravel()

            # Normalizing data if a normalizer is defined
            if svm.normalizer is not None:
                x_norm = svm.normalizer.normalize(x)
                p_norm = svm.normalizer.normalize(p)

            # Testing discriminant_function on multiple points

            df_scores_neg = svm.discriminant_function(x, label=0)
            self.logger.info("discriminant_function(x, label=0):\n"
                             "{:}".format(df_scores_neg))
            _check_df_scores(df_scores_neg, self.dataset.num_samples)

            df_scores_pos = svm.discriminant_function(x, label=1)
            self.logger.info("discriminant_function(x, label=1):\n"
                             "{:}".format(df_scores_pos))
            _check_df_scores(df_scores_pos, self.dataset.num_samples)

            self.assertFalse(
                ((df_scores_pos.sign() * -1) != df_scores_neg.sign()).any())

            # Testing _discriminant_function on multiple points

            ds_priv_scores = svm._discriminant_function(x_norm, label=1)
            self.logger.info("_discriminant_function(x_norm, label=1):\n"
                             "{:}".format(ds_priv_scores))
            _check_df_scores(ds_priv_scores, self.dataset.num_samples)

            # Comparing output of public and private

            self.assertFalse((df_scores_pos != ds_priv_scores).any())

            # Testing classify on multiple points

            labels, scores = svm.classify(self.dataset.X)
            self.logger.info("classify(x):\nlabels: {:}\n"
                             "scores: {:}".format(labels, scores))
            _check_classify_scores(
                labels, scores, self.dataset.num_samples, svm.n_classes)

            # Comparing output of discriminant_function and classify

            self.assertFalse((df_scores_neg != scores[:, 0].ravel()).any())
            self.assertFalse((df_scores_pos != scores[:, 1].ravel()).any())

            # Testing discriminant_function on single point

            df_scores_neg = svm.discriminant_function(p, label=0)
            self.logger.info("discriminant_function(p, label=0):\n"
                             "{:}".format(df_scores_neg))
            _check_df_scores(df_scores_neg, 1)

            df_scores_pos = svm.discriminant_function(p, label=1)
            self.logger.info("discriminant_function(p, label=1):\n"
                             "{:}".format(df_scores_pos))
            _check_df_scores(df_scores_pos, 1)

            self.assertFalse(
                ((df_scores_pos.sign() * -1) != df_scores_neg.sign()).any())

            # Testing _discriminant_function on single point

            df_priv_scores = svm._discriminant_function(p_norm, label=1)
            self.logger.info("_discriminant_function(p_norm, label=1):\n"
                             "{:}".format(df_priv_scores))
            _check_df_scores(df_priv_scores, 1)

            # Comparing output of public and private

            self.assertFalse((df_scores_pos != df_priv_scores).any())

            self.logger.info("Testing classify on single point")

            labels, scores = svm.classify(p)
            self.logger.info("classify(p):\nlabels: {:}\n"
                             "scores: {:}".format(labels, scores))
            _check_classify_scores(labels, scores, 1, svm.n_classes)

            # Comparing output of discriminant_function and classify

            self.assertFalse(
                (df_scores_neg != CArray(scores[:, 0]).ravel()).any())
            self.assertFalse(
                (df_scores_pos != CArray(scores[:, 1]).ravel()).any())

            # Testing error raising

            with self.assertRaises(ValueError):
                svm._discriminant_function(x_norm, label=0)
            with self.assertRaises(ValueError):
                svm._discriminant_function(p_norm, label=0)


if __name__ == '__main__':
    CUnitTest.main()
