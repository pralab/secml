import unittest
from secml.utils import CUnitTest

from secml.figure import CFigure
from secml.array import CArray
from secml.optim.function import CFunction
from secml.core.type_utils import is_scalar


class TestCFunctionRosenbrock(CUnitTest):
    """Unit test for CFunctionRosenbrock."""

    def setUp(self):

        self.fun_obj = CFunction.create('rosenbrock')

        self.logger.info("Global minimum: {:}".format(
            self.fun_obj.global_min()))
        self.logger.info("Global minimum @: {:}".format(
            self.fun_obj.global_min_x()))

    def test_function_result(self):
        """Test if function returns the correct value."""
        x = CArray([-2, 5])

        correct_result = 109

        funct_res = self.fun_obj.fun(x)

        self.logger.info("Correct result: {:}".format(correct_result))
        self.logger.info("Function result: {:}".format(funct_res))

        self.assertTrue(is_scalar(funct_res))
        self.assertEquals(correct_result, funct_res)

        # Testing N-dimensional
        x = CArray([-2, 5, -5])

        correct_result = 90125

        funct_res = self.fun_obj.fun(x)

        self.logger.info("Correct result: {:}".format(correct_result))
        self.logger.info("Function result: {:}".format(funct_res))

        self.assertTrue(is_scalar(funct_res))
        self.assertEquals(correct_result, funct_res)

    def test_2D(self):
        """Plot of a 2D example."""
        grid_limits = [(-1.1, 1.1), (-1.1, 1.1)]

        fig = CFigure(width=7)
        fig.switch_sptype(sp_type='function')
        fig.sp.plot_fobj(func=self.fun_obj.fun, plot_levels=True,
                         grid_limits=grid_limits, levels=[1],
                         n_grid_points=50, n_colors=200, vmin=0, vmax=10)

        fig.sp.title("Rosenbrock Function")
        fig.show()


if __name__ == '__main__':
    unittest.main()
