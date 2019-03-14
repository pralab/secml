from secml.utils import CUnitTest

from secml.array import CArray
from secml.optim.optimizers import COptimizerScipy
from secml.optim.function import CFunction
from secml.optim.constraints import CConstraintBox


class TestCOptimizer(CUnitTest):
    """Test for COptimizer class."""

    def setUp(self):

        avail_funcs = ['3h-camel', 'beale', 'mc-cormick', 'rosenbrock']

        # Instancing the available functions to test optimizer
        self.funcs = {}
        for fun_id in avail_funcs:
            self.funcs[fun_id] = CFunction.create(fun_id)

    def test_minimize(self):
        """Test for COptimizer.minimize() method."""
        self.logger.info("Test for COptimizer.minimize() method.")

        x0 = CArray([1., 0.])  # Starting point for minimization

        for fun_id in self.funcs:
            fun = self.funcs[fun_id]

            self.logger.info(
                "Testing minimization of {:}".format(fun.__class__.__name__))

            opt = COptimizerScipy(fun)

            if fun.class_type == 'mc-cormick':
                # set bounds
                lb,ub = fun.bounds()
                opt.bounds = CConstraintBox(lb,ub)

            min_x = opt.minimize(
                x0, method='L-BFGS-B',
                options={'gtol': 1e-6, 'disp': True})

            self.logger.info("Found minimum: {:}".format(min_x))
            self.logger.info("Fun value @ minimum: {:}".format(opt.f_opt))

            # Round results for easier asserts
            self.assertAlmostEqual(
                round(opt.f_opt, 4), fun.global_min(), places=4)
            self.assertFalse(
                (min_x.round(decimals=4) != fun.global_min_x()).any())

    def test_approx_fprime_check_grad(self):
        """Test if the gradient check made up with
        COptimizer.approx_fprime() and .check_grad() methods is correct."""
        self.logger.info(
            "Test for COptimizer.approx_fprime() and .check_grad() methods.")

        x0 = CArray([1., 0.])  # Starting point for minimization

        for fun_id in self.funcs:
            fun = self.funcs[fun_id]

            self.logger.info(
                "Testing grad approx of {:}".format(fun.__class__.__name__))

            grad_err = fun.check_grad(x0)

            self.logger.info(
                "(Real grad - approx).norm(): {:}".format(grad_err))

            self.assertLess(grad_err, 1e-3)

    # Two dumb function with two required parameters
    def _fun_2_params(self, x, y):
        return 1

    def _dfun_2_params(self, x, y):
        return CArray([0])

    # Two dumb function that have **kwargs as the second parameter
    def _fun_kwargs(self, x, **kwargs):
        if kwargs != {'y': 1}:
            raise ValueError
        return 1

    def _dfun_kwargs(self, x, **kwargs):
        if kwargs != {'y': 1}:
            raise ValueError
        return CArray([0])

    def _fun_args(self, x, *args):
        return self._fun_kwargs(x, **args[0])

    def _dfun_args(self, x, *args):
        return self._dfun_kwargs(x, **args[0])

    def test_approx_fprime_check_param_passage(self):
        """Test the functions COptimizer.approx_fprime() and .check_grad()
        are correctly passing the extra parameters to the main function and
        the one that computes the gradient.

        """
        self.logger.info(
            "Test the parameters passage made up by "
            "COptimizer.approx_fprime() "
            "and .check_grad() methods.")

        x0 = CArray([1.])  # Starting point for minimization
        epsilon = 0.1

        self.logger.info(
            "Testing when the function and the gradient have two parameter")

        fun = CFunction(fun=self._fun_2_params, gradient=self._dfun_2_params)
        self.logger.info("Testing check_grad")

        grad_err = fun.check_grad(x0, 1)
        self.logger.info("Grad error: {:}".format(grad_err))
        self.assertEqual(0, grad_err)

        self.logger.info("Testing approx_fprime")

        grad_err = fun.approx_fprime(x0, epsilon, 1).item()
        self.logger.info("Grad error: {:}".format(grad_err))
        self.assertEqual(0, grad_err)

        self.logger.info(
            "Testing when the function and the gradient have **kwargs "
            " as second parameter")

        fun = CFunction(fun=self._fun_args, gradient=self._dfun_args)

        self.logger.info("Testing check_grad ")

        grad_err = fun.check_grad(x0, {'y': 1})
        self.logger.info("Grad error: {:}".format(grad_err))
        self.assertEqual(0, grad_err)

        self.logger.info("Testing approx_fprime ")

        grad_err = fun.approx_fprime(x0, epsilon, ({'y': 1}))
        self.logger.info("Grad error: {:}".format(grad_err))
        self.assertEqual(0, grad_err)


if __name__ == "__main__":
    CUnitTest.main()
