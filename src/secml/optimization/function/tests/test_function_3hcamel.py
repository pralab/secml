import unittest
from secml.utils import CUnitTest

from secml.figure import CFigure
from secml.array import CArray
from secml.optimization.function import CFunction


class TestCFunctionThreeHumpCamel(CUnitTest):
    """Unit test for CFunctionThreeHumpCamel."""

    def setUp(self):

        self.fun_obj = CFunction.create('3h_camel')

    def test_function_result(self):
        """Test if function returns the correct value."""
        x = CArray([0, 0])

        correct_result = 0

        funct_res = self.fun_obj.fun(x)

        self.logger.info("Correct result: {:}".format(correct_result))
        self.logger.info("Function result: {:}".format(funct_res))

        self.assertEqual(correct_result, funct_res)

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
