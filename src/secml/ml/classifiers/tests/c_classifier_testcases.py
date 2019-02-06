from secml.utils import CUnitTest

from secml.optimization import COptimizer
from secml.optimization.function import CFunction
from secml.core.constants import eps


class CClassifierTestCases(CUnitTest):
    """Unittests interface for CClassifier."""

    def _test_gradient_numerical(self, clf, x, epsilon=eps, **kwargs):
        """Test for clf.gradient_f_x comparing to numerical gradient.

        Parameters
        ----------
        clf : CClassifier
        x : CArray

        """
        def _fun_args(sample, classifier, *f_args):
            return classifier.decision_function(sample, **f_args[0])

        def _grad_args(sample, classifier, *f_args):
            """
            Wrapper needed as gradient_f_x have **kwargs
            """
            return classifier.gradient_f_x(sample, **f_args[0])

        if 'y' in kwargs:
            raise ValueError("`y` cannot be passed to this unittest.")

        for c in clf.classes:

            # TODO: REMOVE AFTER y IS A REQUIRED PARAM OF gradient
            kwargs['y'] = c  # Appending class to test_f_x

            # Analytical gradient
            gradient = clf.gradient_f_x(x, **kwargs)

            self.assertTrue(gradient.is_vector_like)
            self.assertEqual(x.size, gradient.size)

            # Numerical gradient
            num_gradient = COptimizer(
                CFunction(_fun_args, _grad_args)).approx_fprime(
                x, epsilon, clf, kwargs)

            # Compute the norm of the difference
            error = (gradient - num_gradient).norm()

            self.logger.info(
                "Analytic grad wrt. class {:}:\n{:}".format(c, gradient))
            self.logger.info(
                "Numeric gradient wrt. class {:}:\n{:}".format(c, num_gradient))

            self.logger.info("norm(grad - num_grad): {:}".format(error))
            self.assertLess(error, 1e-3)

            for i, elm in enumerate(gradient):
                self.assertIsInstance(elm, float)


if __name__ == '__main__':
    CUnitTest.main()
