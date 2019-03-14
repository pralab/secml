from secml.utils import CUnitTest

from secml.optim.optimizers import CSolverDescDir
from secml.optim.function import CFunction
from secml.optim.constraints import CConstraintL2, CConstraintBox
from secml.array import CArray
from secml.figure import CFigure
from secml.utils import fm


class TestCSolverDescDir(CUnitTest):
    """Unittests for class CSolverDescDir."""

    def setUp(self):
        self.eta = 1e-6
        self.eta_min = 1e-4
        self.eps = 1e-12
        self.filename = 'test_solver_desc_dir'

    def _plot_optimization(self, function_name, solver, grid_limits, g_min,
                           title, vmin=None, vmax=None):
        """Plots the optimization problem."""
        fig = CFigure(markersize=12)
        fig.switch_sptype(sp_type='function')

        # Plot objective function
        fig.sp.plot_fobj(func=CArray.apply_along_axis,
                         plot_background=True,
                         n_grid_points=50, n_colors=50,
                         grid_limits=grid_limits,
                         levels=[1], levels_color='gray', levels_style='--',
                         colorbar=True, func_args=(solver.f.fun, 1,),
                         vmin=vmin, vmax=vmax)

        # Plot distance constraint
        fig.sp.plot_fobj(func=lambda x: solver.constr.constraint(x),
                         plot_background=False, n_grid_points=50,
                         grid_limits=grid_limits, levels=[0], colorbar=False)

        # Plot global minimum
        fig.sp.plot(g_min[0], g_min[1], 'wx',
                    markersize=15, markeredgewidth=2)

        # Plot optimization trace
        fig.sp.plot_path(solver.x_seq)

        fig.sp.title(title)
        fig.savefig(fm.join(fm.abspath(__file__),
                            self.filename + '_' + function_name + '.pdf'),
                    file_format='pdf')

    def test_rosenbrock(self):
        """Test solver on rosenbrock function."""
        fun = CFunction.create('rosenbrock')
        g_min = CArray([1, 1])

        x_init = CArray([-1, -1])

        constr = CConstraintL2(center=x_init, radius=3)
        bound = CConstraintBox(lb=None, ub=None)

        solver = CSolverDescDir(fun, constr=constr, bounds=bound,
                                eta=self.eta, eta_min=self.eta_min,
                                eps=self.eps)
        solver.verbose = 2

        solver.minimize(x_init)

        self.logger.info("Global minimum @ {:}".format(g_min))
        self.logger.info("Found minimum @ {:}".format(solver.x_opt))

        grid_limits = [(-1.1, 1.1), (-1.1, 1.1)]
        title = "Rosenbrock function - " \
                "Global Minimum @ {:}".format(g_min.tolist())
        self._plot_optimization('rosenbrock', solver, grid_limits, g_min,
                                title, vmin=0, vmax=10)

    def test_mccormick(self):
        """Test solver on mccormick function."""
        fun = CFunction.create('mc-cormick')
        g_min = CArray([-0.54719, -1.54719])

        # x_init = CArray([1, 2])  # FIXME: STOPS AT (LOCAL MINIMUM?)
        x_init = CArray([2, 0])

        constr = CConstraintL2(center=x_init, radius=5)
        bound = CConstraintBox(lb=None, ub=None)

        solver = CSolverDescDir(fun, constr=constr, bounds=bound,
                                eta=self.eta, eta_min=self.eta_min,
                                eps=self.eps)
        solver.verbose = 2

        solver.minimize(x_init)

        self.logger.info("Global minimum @ {:}".format(g_min))
        self.logger.info("Found minimum @ {:}".format(solver.x_opt))

        grid_limits = [(-2, 3), (-3, 2)]
        title = "McCormick function - " \
                "Global Minimum @ {:}".format(g_min.tolist())
        self._plot_optimization('mccormick', solver, grid_limits, g_min, title,
                                vmin=-2, vmax=2)

    def test_beale(self):
        """Test solver on Beale function."""
        fun = CFunction.create('beale')
        g_min = CArray([3, 0.5])

        # x_init = CArray([-2, -2])  # FIXME: TOO MANY ITERATIONS, WRONG
        x_init = CArray([0, 0])

        constr = CConstraintL2(center=x_init, radius=6)
        bound = CConstraintBox(lb=None, ub=None)

        solver = CSolverDescDir(fun, constr=constr, bounds=bound,
                                eta=self.eta, eta_min=self.eta_min,
                                eps=self.eps)
        solver.verbose = 2

        solver.minimize(x_init)

        self.logger.info("Global minimum @ {:}".format(g_min))
        self.logger.info("Found minimum @ {:}".format(solver.x_opt))

        grid_limits = [(-4.5, 4.5), (-4.5, 4.5)]
        title = "Beale function - Global Minimum @ {:}".format(g_min.tolist())
        self._plot_optimization('beale', solver, grid_limits, g_min, title,
                                vmin=0, vmax=5)

    def test_3hcamel(self):
        """Test solver on 3HCamel function."""
        fun = CFunction.create('3h-camel')
        g_min = CArray([0, 0])

        x_init = CArray([1, 2])

        constr = CConstraintL2(center=x_init, radius=8)
        bound = CConstraintBox(lb=None, ub=None)

        solver = CSolverDescDir(fun, constr=constr, bounds=bound,
                                eta=self.eta, eta_min=self.eta_min,
                                eps=self.eps)
        solver.verbose = 2

        solver.minimize(x_init)

        self.logger.info("Global minimum @ {:}".format(g_min))
        self.logger.info("Found minimum @ {:}".format(solver.x_opt))

        grid_limits = [(-5, 5), (-5, 5)]
        title = "Three-Hump Camel function - " \
                "Global Minimum @ {:}".format(g_min.tolist())
        self._plot_optimization('3hcamel', solver, grid_limits, g_min, title,
                                vmin=0, vmax=5)


if __name__ == '__main__':
    CUnitTest.main()
