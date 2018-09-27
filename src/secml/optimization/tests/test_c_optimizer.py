from secml.utils import CUnitTest

from secml.optimization.c_optimizer import COptimizer
from secml.optimization.function import CFunction
from secml.array import CArray


class TestCOptimizer(CUnitTest):
    """Test for COptimizer class."""

    def setUp(self):

        avail_funcs = ['3h_camel', 'beale', 'mc_cormick', 'rosenbrock']

        # Instancing the available functions to test optimizer
        self.funcs = {}
        for fun_id in avail_funcs:
            self.funcs[fun_id] = CFunction.create(fun_id)

    def test_minimize(self):
        """Test for COptimizer.minimize() method."""
        self.logger.info("Test for COptimizer.minimize() method.")

        x0 = CArray([0., 0.])  # Starting point for minimization

        for fun_id in self.funcs:

            fun = self.funcs[fun_id]

            self.logger.info(
                "Testing minimization of {:}".format(fun.__class__.__name__))

            opt = COptimizer(fun)
            min_x, jac, fun_val, res = opt.minimize(
                x0, method='BFGS', options={'gtol': 1e-6, 'disp': True})

            self.logger.info("Found minimum: {:}".format(min_x))
            self.logger.info("Fun value @ minimum: {:}".format(fun_val))

            # Round results for easier asserts
            self.assertAlmostEqual(
                round(fun_val, 4), fun.global_min(), places=4)
            self.assertFalse(
                (min_x.round(decimals=4) != fun.global_min_x()).any())

    def test_approx_fprime_check_grad(self):
        """Test for COptimizer.approx_fprime() and .check_grad() methods."""
        self.logger.info(
            "Test for COptimizer.approx_fprime() and .check_grad() methods.")

        x0 = CArray([0., 0.])  # Starting point for minimization

        for fun_id in self.funcs:

            fun = self.funcs[fun_id]

            self.logger.info(
                "Testing grad approx of {:}".format(fun.__class__.__name__))

            opt = COptimizer(fun)
            grad_err = opt.check_grad(x0)

            self.logger.info(
                "(Real grad - approx).norm(): {:}".format(grad_err))

            self.assertLess(grad_err, 1e-3)


if __name__ == "__main__":
    CUnitTest.main()
