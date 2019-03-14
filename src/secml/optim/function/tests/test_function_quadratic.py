import unittest
from secml.utils import CUnitTest

from secml.figure import CFigure
from secml.array import CArray
from secml.optim.function import CFunction
from secml.core.type_utils import is_scalar


class TestCFunctionCircle(CUnitTest):
    """Unit test for CFunction."""

    def setUp(self):
        A = CArray.eye(2, 2)
        b = CArray.zeros((2, 1))
        c = 0
        self.fun_obj = CFunction.create('quadratic', A, b, c)

    def test_function_result(self):
        """Test if function returns the correct value."""
        x = CArray([3, 5])

        correct_result = x[0] ** 2 + x[1] ** 2
        funct_res = self.fun_obj.fun(x)

        self.logger.info("Correct result: {:}".format(correct_result))
        self.logger.info("Actual result: {:}".format(funct_res))

        self.assertTrue(is_scalar(funct_res))
        self.assertEquals(correct_result, funct_res)

    def test_draw_circle(self):
        """Plot of a 2D example."""
        grid_limits = [(-4, 4), (-4, 4)]

        A = CArray.eye(2, 2)
        b = CArray.zeros(2).T
        circle = CFunction.create('quadratic', A, b, 0)

        fig = CFigure()
        fig.switch_sptype(sp_type='function')
        fig.sp.plot_fobj(func=CArray.apply_along_axis, plot_levels=False,
                         grid_limits=grid_limits, func_args=(circle.fun, 1, ))

        fig.sp.title("Quadratic function")
        fig.show()


if __name__ == '__main__':
    unittest.main()
