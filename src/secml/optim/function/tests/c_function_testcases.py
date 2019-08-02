from secml.testing import CUnitTest

from secml.figure import CFigure
from secml.core.type_utils import is_scalar
from secml.utils import fm


class CFunctionTestCases(CUnitTest):
    """Unittests interface for CFunction."""

    def _show_global_min(self, fun):
        """Display the global minimum of the function."""
        self.logger.info("{:}".format(fun.__class__.__name__))
        self.logger.info("Global minimum: {:}".format(fun.global_min()))
        self.logger.info("Global minimum @: {:}".format(fun.global_min_x()))

    def _show_global_max(self, fun):
        """Display the global maximum of the function."""
        self.logger.info("{:}".format(fun.__class__.__name__))
        self.logger.info("Global maximum: {:}".format(fun.global_max()))
        self.logger.info("Global maximum @: {:}".format(fun.global_max_x()))

    def _test_fun_result(self, fun, x, res_expected):
        """Test if function returns the correct value.

        Parameters
        ----------
        fun : CFunction
        x : CArray
        res_expected : scalar

        """
        self.logger.info(
            "Checking value of {:} @ {:}".format(fun.class_type, x))

        res = fun.fun(x)

        self.logger.info("Correct result: {:}".format(res_expected))
        self.logger.info("Function result: {:}".format(res))

        self.assertTrue(is_scalar(res))
        self.assertAlmostEqual(res_expected, res, places=4)

    def _test_2D(self, fun, grid_limits=None, levels=None,
                 vmin=None, vmax=None, fun_args=()):
        """2D plot of the function.

        Parameters
        ----------
        fun : CFunction
        grid_limits : list of tuple or None, optional
        levels : list or None, optional
        vmin, vmax : scalar or None, optional
        fun_args : tuple

        """
        fun_name = fun.__class__.__name__

        self.logger.info("Plotting 2D of {:}".format(fun_name))

        fig = CFigure(width=7)
        fig.sp.plot_fun(func=fun.fun, plot_levels=True,
                        grid_limits=grid_limits, levels=levels,
                        n_grid_points=50, n_colors=200,
                        vmin=vmin, vmax=vmax, func_args=fun_args)

        fig.sp.title(fun_name)
        fig.savefig(fm.join(fm.abspath(__file__),
                            'test_function_{:}.pdf'.format(fun.class_type)))
