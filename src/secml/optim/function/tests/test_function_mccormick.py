from secml.optim.function.tests import CFunctionTestCases

from secml.array import CArray
from secml.optim.function import CFunction


class TestCFunctionMcCormick(CFunctionTestCases):
    """Unit test for CFunctionBeale."""

    def setUp(self):
        self.fun = CFunction.create('mc-cormick')

    def test_fun_result(self):
        """Test if function returns the correct value."""
        self._show_global_min(self.fun)
        self._test_fun_result(
            self.fun, CArray([-0.5472, -1.5472]), self.fun.global_min())

    def test_2D(self):
        """Plot of a 2D example."""
        grid_limits = [(-1.5, 4), (-3, 4)]
        self._test_2D(self.fun, grid_limits, levels=[0], vmin=-2, vmax=2)


if __name__ == '__main__':
    CFunctionTestCases.main()
