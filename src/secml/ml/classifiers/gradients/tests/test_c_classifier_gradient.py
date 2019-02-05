from abc import ABCMeta, abstractmethod, abstractproperty

from secml.utils import CUnitTest
from secml.array import CArray
from secml.optimization import COptimizer
from secml.optimization.function import CFunction


class CClassifierGradientTestCases(object):
    """Wrapper for TestCClassifierGradient to make unittest.main() work correctly."""

    class TestCClassifierGradient(CUnitTest):
        """Unit test for the classifier gradients."""
        __metaclass__ = ABCMeta

        @abstractmethod
        def _dataset_creation(self):
            raise NotImplementedError()

        @abstractproperty
        def clf_list(self):
            raise NotImplementedError()

        @abstractproperty
        def clf_creation_function(self):
            raise NotImplementedError()

        def setUp(self):

            self.seed = 0

            self._dataset_creation()
            self._set_tested_classes()

            self.logger.info("." * 50)
            self.logger.info("Number of Patterns: %s",
                             str(self.dataset.num_samples))
            self.logger.info("Features: %s", str(self.dataset.num_features))

        def _fun_f_args(self, x, *args):
            return self.clf.decision_function(x, **args[0])

        def _grad_f_x_args(self, x, *args):
            """
            Wrapper needed as gradient_f_x have **kwargs
            """
            return self.clf.gradient_f_x(x, **args[0])

        def _clf_gradient_f_x_check(self, clf, clf_idx):

            self.clf = clf

            smpls_idx = CArray.arange(self.dataset.num_samples)
            i = self.dataset.X.randsample(smpls_idx, 1, random_state=self.seed)
            pattern = self.dataset.X[i, :]
            self.logger.info("P {:}: {:}".format(i, pattern))

            for c in self.classes:

                # Compare the analytical grad with the numerical grad
                gradient = clf.gradient_f_x(pattern, y=c)
                num_gradient = COptimizer(
                    CFunction(self._fun_f_args,
                              self._grad_f_x_args)).approx_fprime(pattern,
                                                                  1e-8,
                                                                  ({'y': c}))
                error = (gradient - num_gradient).norm(order=1)
                self.logger.info("Analitic gradient w.r.t. class %s: %s",
                                 str(c), str(gradient))
                self.logger.info("Numeric gradient w.r.t. class %s: %s",
                                 str(c), str(num_gradient))

                self.logger.info(
                    "norm(grad - num_grad): %s", str(error))
                self.assertLess(error, 1e-3, "problematic classifier is " +
                                clf_idx)

                for i, elm in enumerate(gradient):
                    self.assertIsInstance(elm, float)

        def test_f_x_gradient(self):
            """Test the gradient of the classifier discriminant function"""
            self.logger.info(
                "Testing the gradient of the discriminant function")

            normalizer_vals = [False, True]
            combinations_list = [(clf_idx, normalizer) for clf_idx in \
                                 self.clf_list for normalizer in
                                 normalizer_vals]

            for clf_idx, normalizer in combinations_list:
                if normalizer:
                    self.logger.info("Test the {:} classifier when it has "
                                     "a normalizer inside ".format(clf_idx))
                else:
                    self.logger.info("Test the {:} classifier when it does "
                                     "not have a normalizer inside ".format(
                        clf_idx))
                clf = self.clf_creation_function(clf_idx, normalizer)

                clf.fit(self.dataset)
                self._clf_gradient_f_x_check(clf, clf_idx)

        # functionalities to test the gradient of the loss

        def _fun_L_args(self, x, *args):
            """
            Wrapper needed as the loss function have **kwargs
            """
            return self.clf.gradients.L_tot(x, **args[0])

        def _grad_L_params_args(self, x, *args):
            """
            Wrapper needed as the gradient function have **kwargs
            """
            return self.clf.gradients.L_tot_d_params(x, **args[0])

        def _clf_gradient_L_params_check(self, clf, clf_idx):

            self.clf = clf

            smpls_idx = CArray.arange(self.dataset.num_samples)
            i = self.dataset.X.randsample(smpls_idx, 1, random_state=self.seed)
            x = self.dataset.X[i, :]
            y = self.dataset.Y[i]
            self.logger.info("P {:}: x {:}, y {:}".format(i, x, y))

            # Compare the analytical grad with the numerical grad
            if not hasattr(clf, 'gradients'):
                self.logger.info("The computation of the loss fucntion "
                                 "w.r.t. the parameter has not been "
                                 "implmented yet for this classifier")
                return

            gradient = clf.gradients.L_tot_d_params(x, y, clf)
            num_gradient = COptimizer(
                CFunction(self._fun_L_args,
                          self._grad_L_params_args)).approx_fprime(x, 1e-8, (
                {'y': y, 'clf': clf}))
            error = (gradient - num_gradient).norm(order=1)

            self.logger.info("Analitic gradient %s", str(gradient))
            self.logger.info("Numeric gradient %s", str(num_gradient))

            self.logger.info(
                "norm(grad - num_grad): %s", str(error))
            self.assertLess(error, 1e-3, "problematic classifier is " +
                            clf_idx)

            for i, elm in enumerate(gradient):
                self.assertIsInstance(elm, float)

        def test_L_params_gradient(self):
            """Test the gradient of the loss function w.r.t. the classifier
            parameters"""
            self.logger.info(
                "Testing the gradient of the discriminant function")

            normalizer_vals = [False, True]
            combinations_list = [(clf_idx, normalizer) for clf_idx in \
                                 self.clf_list for normalizer in
                                 normalizer_vals]

            for clf_idx, normalizer in combinations_list:
                if normalizer:
                    self.logger.info("Test the {:} classifier when it has "
                                     "a normalizer inside ".format(clf_idx))
                else:
                    self.logger.info("Test the {:} classifier when it does "
                                     "not have a normalizer inside ".format(
                        clf_idx))
                clf = self.clf_creation_function(clf_idx, normalizer)

                clf.store_dual_vars = True
                clf.fit(self.dataset)
                self._clf_gradient_L_params_check(clf, clf_idx)
