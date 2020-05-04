from secml.ml.classifiers.tests import CClassifierTestCases

from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC

from secml.array import CArray
from secml.data.loader import CDLRandom
from secml.ml.classifiers import CClassifierSVM
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVO
from secml.ml.peval.metrics import CMetric
from secml.figure import CFigure


class TestCClassifierMultiOVO(CClassifierTestCases):
    """Unittests for CClassifierMultiOVO."""

    def setUp(self):
        # generate synthetic data
        self.dataset = CDLRandom(n_classes=4, n_clusters_per_class=1).load()

    def test_predict_withsvm(self):

        svc = SVC(kernel='linear', class_weight='balanced')
        multiclass_sklearn = OneVsOneClassifier(svc)
        multiclass = CClassifierMulticlassOVO(classifier=CClassifierSVM,
                                              class_weight='balanced')
        multiclass.verbose = 2

        multiclass.fit(self.dataset.X, self.dataset.Y)
        class_pred, score_pred = multiclass.predict(
            self.dataset.X, return_decision_function=True)

        self.logger.info("Predicted: \n{:}".format(class_pred))
        self.logger.info("Real: \n{:}".format(self.dataset.Y))

        acc = CMetric.create('accuracy').performance_score(
            self.dataset.Y, class_pred)
        self.logger.info("Accuracy: {:}".format(acc))

        multiclass_sklearn.fit(self.dataset.X.get_data(),
                               self.dataset.Y.tondarray())
        y_sklearn = multiclass_sklearn.predict(self.dataset.X.get_data())

        acc_sklearn = CMetric.create('accuracy').performance_score(
            self.dataset.Y, CArray(y_sklearn))
        self.logger.info("Accuracy Sklearn: {:}".format(acc_sklearn))

        self.assertLess(abs(acc - acc_sklearn), 0.21)

    def test_set(self):

        from secml.ml.kernels import CKernelRBF
        multiclass = CClassifierMulticlassOVO(classifier=CClassifierSVM,
                                              C=1, kernel=CKernelRBF())
        # Test set before training
        multiclass.set_params({'C': 100, 'kernel.gamma': 20})
        for clf in multiclass._binary_classifiers:
            self.assertEqual(clf.C, 100.0)
            self.assertEqual(clf.kernel.gamma, 20.0)

        # Restoring kernel
        multiclass.set('kernel', CKernelRBF(gamma=50))

        # Setting different parameter in single trained_classifiers
        multiclass.prepare(num_classes=6)
        different_c = (10, 20, 30, 40, 50, 60)
        multiclass.set('C', different_c)
        different_gamma = (70, 80, 90, 100, 110, 120)
        multiclass.set('kernel.gamma', different_gamma)

        # Fit multiclass classifier than test set after training
        multiclass.fit(self.dataset.X, self.dataset.Y)

        for clf_idx, clf in enumerate(multiclass._binary_classifiers):
            self.assertEqual(clf.C, different_c[clf_idx])
            self.assertEqual(clf.kernel.gamma, different_gamma[clf_idx])

        # Test set after training
        multiclass.set_params({'C': 30, 'kernel.gamma': 200})
        for clf in multiclass._binary_classifiers:
            self.assertEqual(clf.C, 30.0)
            self.assertEqual(clf.kernel.gamma, 200.0)

        for clf in multiclass._binary_classifiers:
            self.assertEqual(clf.C, 30.0)
            self.assertEqual(clf.kernel.gamma, 200.0)

        # Setting parameter in single trained_classifiers
        multiclass._binary_classifiers[0].kernel.gamma = 300
        for i in range(1, multiclass.num_classifiers):
            self.assertNotEqual(
                multiclass._binary_classifiers[i].kernel.gamma, 300.0)

        # Setting different parameter in single trained_classifiers
        different_c = (100, 200, 300)

        # ValueError is raised as not enough binary classifiers are available
        with self.assertRaises(ValueError):
            multiclass.set('C', different_c)

        multiclass.prepare(num_classes=3)
        multiclass.set('C', different_c)
        for clf_idx, clf in enumerate(multiclass._binary_classifiers):
            self.assertEqual(clf.C, different_c[clf_idx])

    def test_apply_method(self):

        multiclass = CClassifierMulticlassOVO(classifier=CClassifierSVM,
                                              class_weight='balanced')
        multiclass.fit(self.dataset.X,self.dataset.Y)
        multiclass.apply_method(CClassifierSVM.set, param_name='C',
                                param_value=150)

        for i in range(multiclass.num_classifiers):
            self.assertEqual(multiclass._binary_classifiers[i].C, 150)

    def test_normalization(self):
        """Test data normalization inside CClassifierMulticlassOVO."""
        from secml.ml.features.normalization import CNormalizerMinMax

        ds_norm_x = CNormalizerMinMax().fit_transform(self.dataset.X)

        multi_nonorm = CClassifierMulticlassOVO(classifier=CClassifierSVM,
                                                class_weight='balanced')
        multi_nonorm.fit(ds_norm_x, self.dataset.Y)
        pred_y_nonorm = multi_nonorm.predict(ds_norm_x)

        multi = CClassifierMulticlassOVO(classifier=CClassifierSVM,
                                         class_weight='balanced',
                                         preprocess='min-max')
        multi.fit(self.dataset.X, self.dataset.Y)
        pred_y = multi.predict(self.dataset.X)

        self.logger.info(
            "Predictions with internal norm:\n{:}".format(pred_y))
        self.logger.info(
            "Predictions with external norm:\n{:}".format(pred_y_nonorm))

        self.assertFalse((pred_y_nonorm != pred_y).any())

    def test_plot_decision_function(self):
        """Test plot of multiclass classifier decision function."""
        # generate synthetic data
        ds = CDLRandom(n_classes=3, n_features=2, n_redundant=0,
                       n_clusters_per_class=1, class_sep=1,
                       random_state=0).load()

        multiclass = CClassifierMulticlassOVO(
            classifier=CClassifierSVM,
            class_weight='balanced',
            preprocess='min-max')

        # Training and classification
        multiclass.fit(ds.X, ds.Y)
        y_pred, score_pred = multiclass.predict(
            ds.X, return_decision_function=True)

        def plot_hyperplane(img, clf, min_v, max_v, linestyle, label):
            """Plot the hyperplane associated to the OVO clf."""
            xx = CArray.linspace(
                min_v - 5, max_v + 5)  # make sure the line is long enough
            # get the separating hyperplane
            yy = -(clf.w[0] * xx + clf.b) / clf.w[1]
            img.sp.plot(xx, yy, linestyle, label=label)

        fig = CFigure(height=7, width=8)
        fig.sp.title('{:} ({:})'.format(multiclass.__class__.__name__,
                                        multiclass.classifier.__name__))

        x_bounds, y_bounds = ds.get_bounds()

        styles = ['go-', 'yp--', 'rs-.', 'bD--', 'c-.', 'm-', 'y-.']

        for c_idx, c in enumerate(ds.classes):
            # Plot boundary and predicted label for each OVO classifier

            plot_hyperplane(fig, multiclass._binary_classifiers[c_idx],
                            x_bounds[0], x_bounds[1], styles[c_idx],
                            'Boundary\nfor class {:}'.format(c))

            fig.sp.scatter(ds.X[ds.Y == c, 0],
                           ds.X[ds.Y == c, 1],
                           s=40, c=styles[c_idx][0])
            fig.sp.scatter(ds.X[y_pred == c, 0], ds.X[y_pred == c, 1], s=160,
                           edgecolors=styles[c_idx][0],
                           facecolors='none', linewidths=2)

        # Plotting multiclass decision function
        fig.sp.plot_decision_regions(multiclass, n_grid_points=100,
                                     grid_limits=ds.get_bounds(offset=5))

        fig.sp.xlim(x_bounds[0] - .5 * x_bounds[1],
                    x_bounds[1] + .5 * x_bounds[1])
        fig.sp.ylim(y_bounds[0] - .5 * y_bounds[1],
                    y_bounds[1] + .5 * y_bounds[1])

        fig.sp.legend(loc=4)  # lower, right

        fig.show()

    def test_fun(self):
        """Test for decision_function() and predict() methods."""
        self.logger.info(
            "Test for decision_function() and predict() methods.")

        mc = CClassifierMulticlassOVO(classifier=CClassifierSVM,
                                      class_weight='balanced')

        scores_d = self._test_fun(mc, self.dataset.todense())
        scores_s = self._test_fun(mc, self.dataset.tosparse())

        self.assert_array_almost_equal(scores_d, scores_s)

    def test_gradient(self):
        """Unittests for gradient() function."""
        multiclass = CClassifierMulticlassOVO(classifier=CClassifierSVM,
                                              class_weight='balanced')

        i = 5  # Sample to test

        self.logger.info("Testing with dense data...")
        ds = self.dataset.todense()
        multiclass.fit(ds.X, ds.Y)

        pattern = ds.X[i, :]

        # Check if we can return the i_th classifier
        for i in range(multiclass.n_classes):

            # Compute the gradient for class i
            ovo_grad_pos = CArray.zeros(
                shape=pattern.shape, dtype=pattern.dtype, sparse=pattern.issparse)
            ovo_grad_neg = CArray.zeros(
                shape=pattern.shape, dtype=pattern.dtype, sparse=pattern.issparse)
            for j in range(multiclass.num_classifiers):
                idx_pos = multiclass._clf_pair_idx[j][0]
                idx_neg = multiclass._clf_pair_idx[j][1]

                if idx_pos == i:
                    w_bin = CArray([1, 0])
                    grad_pos = \
                        multiclass._binary_classifiers[j].gradient(pattern, w_bin)
                    ovo_grad_pos += grad_pos
                if idx_neg == i:
                    w_bin = CArray([0, 1])
                    grad_neg = \
                        multiclass._binary_classifiers[j].gradient(pattern, w_bin)
                    ovo_grad_neg += grad_neg

            ovo_grad = (ovo_grad_pos + ovo_grad_neg) / 3

            w = CArray.zeros(shape=multiclass.n_classes)
            w[i] = 1  # one-hot encoding of y
            gradient = multiclass.gradient(pattern, w)
            self.logger.info(
                "Gradient of {:}^th sub-clf is:\n{:}".format(i, gradient))

            self.assert_array_almost_equal(gradient.atleast_2d(), -ovo_grad)

        self.logger.info("Testing with sparse data...")
        ds = self.dataset.tosparse()
        multiclass.fit(ds.X, ds.Y)

        pattern = ds.X[i, :]

        # Compare dense gradients with sparse gradients
        grads_d = self._test_gradient_numerical(multiclass, pattern)
        grads_s = self._test_gradient_numerical(multiclass, pattern)

        for grad_i, grad in enumerate(grads_d):
            self.assert_array_almost_equal(grad.atleast_2d(), grads_s[grad_i])

        # Test error raise
        # TODO: Change grad_f_x with gradient after checking clf_idx in gradient(x,w)
        with self.assertRaises(ValueError):
            multiclass.grad_f_x(pattern, y=-1)
        with self.assertRaises(ValueError):
            multiclass.grad_f_x(pattern, y=100)

    def test_multiclass_gradient(self):
        """Test if gradient is correct when requesting for all classes with w"""

        multiclass = CClassifierMulticlassOVO(classifier=CClassifierSVM,
                                              class_weight='balanced')
        multiclass.fit(self.dataset.X, self.dataset.Y)
        div = CArray.rand(shape=multiclass.n_classes, random_state=0)

        def f_x(z):
            z = multiclass.predict(z, return_decision_function=True)[1]
            return CArray((z / div).mean())

        def grad_f_x(p):
            w = CArray.ones(shape=multiclass.n_classes) / \
                (div * multiclass.n_classes)
            return multiclass.gradient(p, w=w)

        i = 5  # Sample to test
        x = self.dataset.X[i, :]

        from secml.optim.function import CFunction
        check_grad_val = CFunction(f_x, grad_f_x).check_grad(x, epsilon=1e-1)
        self.logger.info(
            "norm(grad - num_grad): %s", str(check_grad_val))
        self.assertLess(check_grad_val, 1e-3)

    def test_preprocess(self):
        """Test classifier with preprocessors inside."""
        multiclass = CClassifierMulticlassOVO(classifier=CClassifierSVM,
                                              class_weight='balanced')

        # All linear transformations with gradient implemented
        self._test_preprocess(self.dataset, multiclass,
                              ['min-max', 'mean-std'],
                              [{'feature_range': (-1, 1)}, {}])
        self._test_preprocess_grad(self.dataset, multiclass,
                                   ['min-max', 'mean-std'],
                                   [{'feature_range': (-1, 1)}, {}])

        # Mixed linear/nonlinear transformations without gradient
        self._test_preprocess(
            self.dataset, multiclass, ['pca', 'unit-norm'], [{}, {}])
