from secml.optim.function.tests import CFunctionTestCases

from secml.array import CArray
from secml.optim.function import CFunction


class TestCFunctionBeale(CFunctionTestCases):
    """Unit test for CFunctionBeale."""

    def setUp(self):
        self.fun = CFunction.create('beale')

    def test_fun_result(self):
        """Test if function returns the correct value."""
        self._show_global_min(self.fun)
        self._test_fun_result(
            self.fun, CArray([3, 0.5]), self.fun.global_min())

    def test_2D(self):
        """Plot of a 2D example."""
        grid_limits = [(-4.5, 4.5), (-4.5, 4.5)]
        self._test_2D(self.fun, grid_limits, levels=[1], vmin=0, vmax=5)


if __name__ == '__main__':
    CFunctionTestCases.main()
