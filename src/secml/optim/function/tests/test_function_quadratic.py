from secml.optim.function.tests import CFunctionTestCases

from secml.array import CArray
from secml.optim.function import CFunction


class TestCFunctionCircle(CFunctionTestCases):
    """Unit test for CFunctionQuadratic."""

    def setUp(self):
        A = CArray.eye(2, 2)
        b = CArray.zeros((2, 1))
        c = 0
        self.fun = CFunction.create('quadratic', A, b, c)

    def test_fun_result(self):
        """Test if function returns the correct value."""
        x = CArray([3, 5])
        correct_result = x[0] ** 2 + x[1] ** 2
        self._test_fun_result(self.fun, x, correct_result.item())

    def test_2D(self):
        """Plot of a 2D example."""
        grid_limits = [(-4, 4), (-4, 4)]

        A = CArray.eye(2, 2)
        b = CArray.zeros(2).T
        circle = CFunction.create('quadratic', A, b, 0)

        self._test_2D(circle, grid_limits, levels=[16])


if __name__ == '__main__':
    CFunctionTestCases.main()
