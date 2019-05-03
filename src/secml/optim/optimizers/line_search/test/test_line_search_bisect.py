import unittest
from secml.testing import CUnitTest
from secml.optim.function import CFunction
from secml.array import CArray
from secml.figure import CFigure
from secml.optim.optimizers.line_search import CLineSearchBisect
from secml.utils import fm


class TestLineSearch(CUnitTest):
    """Test for COptimizer class."""

    def test_minimize(self):
        """Testing the bisect line-search algorithm."""
        self.logger.info(
            "Test for binary line search  ... ")

        def fun_test(x):
            return x ** 2 - 1

        self.fun = CFunction(fun=fun_test)

        line_search = CLineSearchBisect(fun=self.fun, max_iter=40)
        line_search.verbose = 2

        x = CArray([-5.0])
        fx = self.fun.fun(x)
        self.logger.info("x: " + str(x))
        self.logger.info("f(x): " + str(fx))

        d = CArray([1.0])
        x0, f0 = line_search.minimize(x, d, fx=fx)

        self.logger.info("x*: " + str(x0))
        self.logger.info("f(x*): " + str(f0))
        self.logger.info("num. iter.: " + str(line_search.n_iter))
        self.logger.info("num. eval.: " + str(self.fun.n_fun_eval))

        self._save_fig()

        self.assertTrue(x0.norm() <= 1e-6,
                        "Correct solution found, x0 = 0.")

    def _save_fig(self):
        """Visualizing the function being optimized with line search."""
        x_range = CArray.arange(-5, 20, 0.5, )
        score_range = x_range.T.apply_along_axis(self.fun.fun, axis=1)
        ref_line = CArray.zeros(x_range.size)
        fig = CFigure(height=6, width=12)
        fig.sp.plot(x_range, score_range, color='b')
        fig.sp.plot(x_range, ref_line, color='k')
        filename = fm.join(fm.abspath(__file__), 'test_line_search_bisect.pdf')
        fig.savefig(filename)


if __name__ == "__main__":
    unittest.main()
