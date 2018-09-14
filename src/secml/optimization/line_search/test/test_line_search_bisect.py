import unittest
from secml.utils import CUnitTest
from secml.optimization.function import CFunction
from secml.array import CArray
from secml.figure import CFigure
from secml.optimization.line_search import CLineSearchBisect


class TestLineSearch(CUnitTest):
    """Test for COptimizer class."""

    def test_minimize(self):
        """Testing function line search."""
        self.logger.info(
            "Test for binary line search  ... ")

        def fun_test(x):
            return x.ravel() ** 2 - 1

        self.fun = CFunction(fun=fun_test)

        line_search = CLineSearchBisect(fun=self.fun)
        line_search.verbose = 2

        x = CArray([-1.0])
        d = CArray([+1.0])
        x0, f0 = line_search.line_search(x, d)

        self.logger.info("x*: " + str(x0))
        self.logger.info("f(x*): " + str(f0))
        self.logger.info("num. iter.: " + str(line_search.n_iter))
        self.logger.info("num. eval.: " + str(self.fun.n_fun_eval))

        self._plot_fun()

        self.assertTrue(x0.norm() <= 1e-6,
                        "Correct solution found, x0 = 0.")

    def _plot_fun(self):
        # cons_box = CConstraintBox(ub=3, lb=-3)
        x_range = CArray.arange(-5, 20, 0.5, )
        score_range = self.fun.fun(x_range)
        # self.logger.info("Result scores : " + str(score_range))
        ref_line = CArray.zeros(x_range.size)
        fig = CFigure(height=6, width=12)
        fig.sp.plot(x_range, score_range, color='b')
        fig.sp.plot(x_range, ref_line, color='k')
        fig.show()


if __name__ == "__main__":
    unittest.main()
