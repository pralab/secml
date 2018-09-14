import unittest
from secml.utils import CUnitTest

from secml.figure import CFigure
from secml.array import CArray
from secml.optimization.function import CFunction


class TestCFunctionMcCormick(CUnitTest):
    """Unit test for CFunctionBeale."""

    def setUp(self):

        self.fun_obj = CFunction.create('mc_cormick')

    def test_function_result(self):
        """Test if function returns the correct value."""
        x = CArray([-0.54719, -1.54719])

        correct_result = -1.9133

        funct_res = self.fun_obj.fun(x)

        self.logger.info("Correct result: {:}".format(correct_result))
        self.logger.info("Function result: {:}".format(funct_res))

        self.assertAlmostEqual(correct_result, funct_res, places=3)

    def test_2D(self):
        """Plot of a 2D example."""
        grid_limits = [(-1.5, 4), (-3, 4)]

        fig = CFigure(width=7)
        fig.switch_sptype(sp_type='function')
        fig.sp.plot_fobj(func=self.fun_obj.fun, plot_levels=True,
                         grid_limits=grid_limits, levels=[0],
                         n_grid_points=50, n_colors=200, vmin=-2, vmax=2)

        fig.sp.title("McCormick Function")
        fig.show()


if __name__ == '__main__':
    unittest.main()
