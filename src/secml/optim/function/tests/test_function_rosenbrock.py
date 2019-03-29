from secml.optim.function.tests import CFunctionTestCases

from secml.array import CArray
from secml.optim.function import CFunction


class TestCFunctionRosenbrock(CFunctionTestCases):
    """Unit test for CFunctionRosenbrock."""

    def setUp(self):
        self.fun = CFunction.create('rosenbrock')

    def test_fun_result(self):
        """Test if function returns the correct value."""
        self._show_global_min(self.fun)
        self._test_fun_result(self.fun, CArray([-2, 5]), 109)
        # Testing N-dimensional
        self._test_fun_result(self.fun, CArray([-2, 5, -5]), 90125)

    def test_2D(self):
        """Plot of a 2D example."""
        grid_limits = [(-1.1, 1.1), (-1.1, 1.1)]
        self._test_2D(self.fun, grid_limits, levels=[1], vmin=0, vmax=10)


if __name__ == '__main__':
    CFunctionTestCases.main()
