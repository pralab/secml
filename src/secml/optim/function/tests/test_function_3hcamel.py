from secml.optim.function.tests import CFunctionTestCases

from secml.array import CArray
from secml.optim.function import CFunction


class TestCFunctionThreeHumpCamel(CFunctionTestCases):
    """Unit test for CFunctionThreeHumpCamel."""

    def setUp(self):
        self.fun = CFunction.create('3h-camel')

    def test_fun_result(self):
        """Test if function returns the correct value."""
        self._show_global_min(self.fun)
        self._test_fun_result(
            self.fun, CArray([0, 0]), self.fun.global_min())

    def test_2D(self):
        """Plot of a 2D example."""
        grid_limits = [(-5, 5), (-5, 5)]
        self._test_2D(self.fun, grid_limits, levels=[1], vmin=0, vmax=5)


if __name__ == '__main__':
    CFunctionTestCases.main()
