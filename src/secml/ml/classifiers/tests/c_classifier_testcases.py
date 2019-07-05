from secml.testing import CUnitTest

from secml.data import CDataset
from secml.ml.features import CPreProcess
from secml.optim.function import CFunction
from secml.core.constants import eps


class CClassifierTestCases(CUnitTest):
    """Unittests interface for CClassifier."""

    def _test_gradient_numerical(self, clf, x, extra_classes=None,
                                 th=1e-3, epsilon=eps, **grad_kwargs):
        """Test for clf.grad_f_x comparing to numerical gradient.

        Parameters
        ----------
        clf : CClassifier
        x : CArray
        extra_classes : None or list of int, optional
            Any extra class which gradient wrt should be tested
        th : float, optional
            The threshold for the check with numerical gradient.
        epsilon : float, optional
            The epsilon to use for computing the numerical gradient.
        grad_kwargs : kwargs
            Any extra parameter for the gradient function.

        """
        if 'y' in grad_kwargs:
            raise ValueError("`y` cannot be passed to this unittest.")

        if extra_classes is not None:
            classes = clf.classes.append(extra_classes)
        else:
            classes = clf.classes

        for c in classes:

            grad_kwargs['y'] = c  # Appending class to test_f_x

            # Analytical gradient
            gradient = clf.grad_f_x(x, **grad_kwargs)

            self.assertTrue(gradient.is_vector_like)
            self.assertEqual(x.size, gradient.size)

            # Numerical gradient
            num_gradient = CFunction(
                clf.decision_function).approx_fprime(x, epsilon, y=c)

            # Compute the norm of the difference
            error = (gradient - num_gradient).norm()

            self.logger.info(
                "Analytic grad wrt. class {:}:\n{:}".format(c, gradient))
            self.logger.info(
                "Numeric gradient wrt. class {:}:\n{:}".format(
                    c, num_gradient))

            self.logger.info("norm(grad - num_grad): {:}".format(error))
            self.assertLess(error, th)

            self.assertIsSubDtype(gradient.dtype, float)

    @staticmethod
    def _create_preprocess_chain(pre_id_list, kwargs_list):
        """Creates a preprocessor with other preprocessors chained
        and a list of the same preprocessors (not chained)"""
        chain = None
        pre_list = []
        for i, pre_id in enumerate(pre_id_list):
            chain = CPreProcess.create(
                pre_id, preprocess=chain, **kwargs_list[i])
            pre_list.append(CPreProcess.create(pre_id, **kwargs_list[i]))

        return chain, pre_list

    def _create_preprocess_test(self, ds, clf, pre_id_list, kwargs_list):
        """Fit 2 clf, one with internal preprocessor chain
        and another using pre-transformed data.

        Parameters
        ----------
        ds : CDataset
        clf : CClassifier
        pre_id_list : list of str
            This list should contain the class_id of each preprocessor
            that should be part of the chain.
        kwargs_list : list of dict
            This list should contain a dictionary of extra parameter for
            each preprocessor that should be part of the chain.

        Returns
        -------
        pre1 : CPreProcess
            The preprocessors chain.
        data_pre : CArray
            Data (ds.X) transformed using pre1.
        clf_pre : CClassifier
            The classifier with a copy the preprocessors chain inside,
            trained on ds.
        clf : CClassifier
            The classifier without the preprocessors chain inside,
            trained on data_pre.

        """
        pre1 = CPreProcess.create_chain(pre_id_list, kwargs_list)
        data_pre = pre1.fit_transform(ds.X)

        pre2 = CPreProcess.create_chain(pre_id_list, kwargs_list)
        clf_pre = clf.deepcopy()
        clf_pre.preprocess = pre2

        clf_pre.fit(ds)
        clf.fit(CDataset(data_pre, ds.Y))

        return pre1, data_pre, clf_pre, clf

    def _test_preprocess(self, ds, clf, pre_id_list, kwargs_list):
        """Test if clf with preprocessor inside returns the same
        prediction of the clf trained on pre-transformed data.

        Parameters
        ----------
        ds : CDataset
        clf : CClassifier
        pre_id_list : list of str
            This list should contain the class_id of each preprocessor
            that should be part of the chain.
        kwargs_list : list of dict
            This list should contain a dictionary of extra parameter for
            each preprocessor that should be part of the chain.

        """
        pre, data_pre, clf_pre, clf = self._create_preprocess_test(
            ds, clf, pre_id_list, kwargs_list)

        self.logger.info(
            "Testing clf with preprocessor inside:\n{:}".format(clf_pre))

        y1, score1 = clf_pre.predict(ds.X, return_decision_function=True)
        y2, score2 = clf.predict(data_pre, return_decision_function=True)

        self.assert_array_equal(y1, y2)
        self.assert_array_almost_equal(score1, score2)

        # The number of features of the clf with preprocess inside should be
        # equal to the number of dataset features (so before preprocessing)
        self.assertEqual(ds.num_features, clf_pre.n_features)

    def _test_preprocess_grad(self, ds, clf, pre_id_list, kwargs_list,
                              extra_classes=None, check_numerical=True,
                              th=1e-3, epsilon=eps, **grad_kwargs):
        """Test if clf gradient with preprocessor inside is equal to the
        gradient of the clf trained on pre-transformed data.
        Also compare the gradient of the clf with preprocessor
        inside with numerical gradient.

        Parameters
        ----------
        ds : CDataset
        clf : CClassifier
        pre_id_list : list of str
            This list should contain the class_id of each preprocessor
            that should be part of the chain.
        kwargs_list : list of dict
            This list should contain a dictionary of extra parameter for
            each preprocessor that should be part of the chain.
        extra_classes : None or list of int, optional
            Any extra class which gradient wrt should be tested
        check_numerical : bool, optional
            If True, the gradient will be compared with
            the numerical approximation.
        th : float, optional
            The threshold for the check with numerical gradient.
        epsilon : float, optional
            The epsilon to use for computing the numerical gradient.
        grad_kwargs : kwargs
            Any extra parameter for the gradient function.

        """
        pre, data_pre, clf_pre, clf = self._create_preprocess_test(
            ds, clf, pre_id_list, kwargs_list)

        self.logger.info("Testing clf gradient with preprocessor "
                         "inside:\n{:}".format(clf_pre))

        if 'y' in grad_kwargs:
            raise ValueError("`y` cannot be passed to this unittest.")

        if extra_classes is not None:
            classes = clf.classes.append(extra_classes)
        else:
            classes = clf.classes

        for c in classes:

            self.logger.info(
                "Testing grad wrt. class {:}".format(c))

            # Grad of clf without preprocessor inside (using transformed data)
            v_pre = data_pre[0, :]
            clf_grad = clf.grad_f_x(v_pre, y=c, **grad_kwargs)

            # Output of grad_f_x should be a float vector
            self.assertEqual(1, clf_grad.ndim)
            self.assertIsSubDtype(clf_grad.dtype, float)

            # Gradient of clf with preprocessor inside
            v = ds.X[0, :]
            clf_pre_grad = clf_pre.grad_f_x(v, y=c, **grad_kwargs)

            # Gradient of the preprocessor. Should be equal to the gradient
            # of the clf with preprocessor inside
            pre_grad = pre.gradient(v_pre, w=clf_grad)

            # As clf_grad should be a float vector,
            # output of gradient should be the same
            self.assertEqual(1, pre_grad.ndim)
            self.assertIsSubDtype(pre_grad.dtype, float)

            self.assert_array_almost_equal(clf_pre_grad, pre_grad)

        if check_numerical is True:
            # Comparison with numerical gradient
            self._test_gradient_numerical(
                clf_pre, ds.X[0, :], extra_classes=extra_classes,
                th=th, epsilon=epsilon, **grad_kwargs)


if __name__ == '__main__':
    CUnitTest.main()
