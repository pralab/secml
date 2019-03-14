import unittest
from secml.utils import CUnitTest

from secml.figure import CFigure
from secml.array import CArray
from secml.optim.function import CFunction
from secml.core.type_utils import is_scalar


class TestCFunctionThreeHumpCamel(CUnitTest):
    """Unit test for CFunctionThreeHumpCamel."""

    def setUp(self):

        self.fun_obj = CFunction.create('3h-camel')

        self.logger.info("Global minimum: {:}".format(
            self.fun_obj.global_min()))
        self.logger.info("Global minimum @: {:}".format(
            self.fun_obj.global_min_x()))

    def test_function_result(self):
        """Test if function returns the correct value."""
        x = CArray([0, 0])

        funct_res = self.fun_obj.fun(x)

        self.logger.info(
            "Correct result: {:}".format(self.fun_obj.global_min()))
        self.logger.info("Function result: {:}".format(funct_res))

        self.assertTrue(is_scalar(funct_res))
        self.assertAlmostEqual(self.fun_obj.global_min(), funct_res, places=4)

    def test_2D(self):
        """Plot of a 2D example."""
        grid_limits = [(-5, 5), (-5, 5)]

        fig = CFigure(width=7)
        fig.switch_sptype(sp_type='function')
        fig.sp.plot_fobj(func=self.fun_obj.fun, plot_levels=True,
                         grid_limits=grid_limits, levels=[1],
                         n_grid_points=50, n_colors=200, vmin=0, vmax=5,)

        fig.sp.title("ThreeHumpCamel Function")
        fig.show()


if __name__ == '__main__':
    unittest.main()
