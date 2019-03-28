from secml.testing import CUnitTest

from secml.array import CArray
from secml.optim.function import CFunction
from secml.optim.constraints import CConstraint
from secml.figure import CFigure
import secml.utils.c_file_manager as fm


class COptimizerTestCases(CUnitTest):
    """Unittests interface for COptimizer."""

    def setUp(self):

        self.test_funcs = dict()

        # Instancing the available functions to test optimizer
        self.test_funcs['3h-camel'] = {
            'fun': CFunction.create('3h-camel'),
            'x0': CArray([1, 1]),
            'grid_limits': [(-2, 2), (-2, 2)],
            'vmin': 0, 'vmax': 5
        }
        self.test_funcs['beale'] = {
            'fun': CFunction.create('beale'),
            'x0': CArray([0, 0]),
            'grid_limits': [(-1, 4.5), (-1, 1.5)],
            'vmin': 0, 'vmax': 5
        }
        self.test_funcs['mc-cormick'] = {
            'fun': CFunction.create('mc-cormick'),
            'x0': CArray([0, 1]),
            'grid_limits': [(-2, 3), (-3, 1)],
            'vmin': -2, 'vmax': 2
        }
        self.test_funcs['rosenbrock'] = {
            'fun': CFunction.create('rosenbrock'),
            'x0': CArray([-1, -1]),
            'grid_limits': [(-1.1, 1.1), (-1.1, 1.1)],
            'vmin': 0, 'vmax': 10
        }
        quad = self._create_quadratic_fun()
        self.test_funcs['quadratic'] = {
            'fun': quad,
            'x0': CArray([4., -4.]),
            'grid_limits': [(-5, 5), (-5, 5)],
            'vmin': None, 'vmax': None
        }

    def _create_quadratic_fun(self):
        """Creates a quadratic function."""
        A = CArray.eye(2, 2)
        b = CArray.zeros((2, 1))
        c = 0

        discr_fun = CFunction.create('quadratic', A, b, c)
        discr_fun.global_min = lambda: 0.
        discr_fun.global_min_x = lambda: CArray([0, 0])

        return discr_fun

    def _test_minimize(self, opt_class, fun_id,
                       opt_params=None, minimize_params=None, label=None):
        """Test for COptimizer.minimize() method.

        Parameters
        ----------
        opt_class : class
            COptimizer.
        fun_id : str
            Id of the function to test. Should be available in "test_funcs"
            dictionary (see setUp).
        opt_params : dict or None, optional
            Dictionary of parameters for the optimizer.
        minimize_params : dict or None, optional
            Dictionary of parameters for the minimize method.
        label : str or None, optional
            Label to identify the test.

        """
        minimize_params = {} if minimize_params is None else minimize_params
        opt_params = {} if opt_params is None else opt_params

        fun_dict = self.test_funcs[fun_id]

        fun = fun_dict['fun']

        opt = opt_class(fun, **opt_params)
        opt.verbose = 1

        self.logger.info(
            "Testing minimization of {:} using {:}".format(
                fun.__class__.__name__, opt.__class__.__name__))

        if fun.class_type == 'mc-cormick' and 'bounds' not in opt_params:
            self.logger.info("Setting default bounds for mc-cormick function.")
            # set default bounds
            opt.bounds = CConstraint.create('box', *fun.bounds())

        min_x = opt.minimize(fun_dict['x0'], **minimize_params)

        self.logger.info("x0: {:}".format(fun_dict['x0']))
        self.logger.info("Found minimum: {:}".format(min_x))
        self.logger.info("Fun value @ minimum: {:}".format(opt.f_opt))

        self._plot_optimization(opt, fun_dict['x0'], min_x,
                                grid_limits=fun_dict['grid_limits'],
                                method=minimize_params.get('method'),
                                vmin=fun_dict['vmin'],
                                vmax=fun_dict['vmax'],
                                label=label)

        # Round results for easier asserts
        self.assertAlmostEqual(opt.f_opt, fun.global_min(), places=4)
        self.assert_array_almost_equal(
            min_x, fun.global_min_x(), decimal=4)

    def _plot_optimization(
            self, solver, x_0, g_min, grid_limits,
            method=None, vmin=None, vmax=None, label=None):
        """Plots the optimization problem.

        Parameters
        ----------
        solver : COptimizer
        x_0 : CArray
            Starting point.
        g_min : CArray
            Final point (after optimization).
        grid_limits : list of tuple
        vmin, vmax : int or None, optional
        label : str or None, optional

        """
        fig = CFigure(markersize=12)
        fig.switch_sptype(sp_type='function')

        # Plot objective function
        fig.sp.plot_fobj(func=CArray.apply_along_axis,
                         plot_background=True,
                         n_grid_points=30, n_colors=25,
                         grid_limits=grid_limits,
                         levels=[1], levels_color='gray', levels_style='--',
                         colorbar=True, func_args=(solver.f.fun, 1,),
                         vmin=vmin, vmax=vmax)

        if solver.bounds is not None:  # Plot box constraint
            fig.sp.plot_fobj(func=lambda x: solver.bounds.constraint(x),
                             plot_background=False, n_grid_points=20,
                             grid_limits=grid_limits, levels=[0],
                             colorbar=False)

        if solver.constr is not None:  # Plot distance constraint
            fig.sp.plot_fobj(func=lambda x: solver.constr.constraint(x),
                             plot_background=False, n_grid_points=20,
                             grid_limits=grid_limits, levels=[0],
                             colorbar=False)

        # Plot optimization trace
        if solver.x_seq is not None:
            fig.sp.plot_path(solver.x_seq)
        else:
            fig.sp.plot_path(x_0.append(g_min, axis=0))

        fig.sp.title("{:}(fun={:}) - Glob Min @ {:}".format(
            solver.class_type, solver.f.class_type,
            solver.f.global_min_x().tolist()))

        if method is None:
            filename = fm.join(
                fm.abspath(__file__),
                solver.class_type + '-' + solver.f.class_type)
        else:
            filename = fm.join(
                fm.abspath(__file__),
                solver.class_type + '-' + method + '-' + solver.f.class_type)

        filename += '-' + label if label is not None else ''

        fig.savefig(filename + '.pdf')
