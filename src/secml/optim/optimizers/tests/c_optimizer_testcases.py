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
            'vmin': 0, 'vmax': 1
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
            'grid_limits': [(-2.1, 1.1), (-2.1, 1.1)],
            'vmin': 0, 'vmax': 10
        }
        quad = self._quadratic_fun(2)
        self.test_funcs['quad-2'] = {
            'fun': quad,
            'x0': CArray([4, -4]),
            'grid_limits': [(-5, 5), (-5, 5)],
            'vmin': None, 'vmax': None
        }
        n = 100
        quad = self._quadratic_fun(n)
        self.test_funcs['quad-100-sparse'] = {
            'fun': quad,
            'x0': CArray.zeros((n,), dtype=int).tosparse(dtype=int),
        }
        n = 2
        poly = self._create_poly(d=n)
        self.test_funcs['poly-2'] = {
            'fun': poly,
            'x0': CArray.ones((n,)) * 2,
            'vmin': -10, 'vmax': 5,
            'grid_limits': [(-1, 1), (-1, 1)]
        }
        n = 100
        # x0 is a sparse CArray and the solution is a zero vector
        poly = self._create_poly(d=n)
        self.test_funcs['poly-100-int'] = {
            'fun': poly,
            'x0': CArray.ones((n,), dtype=int) * 2
        }
        n = 100
        poly = self._create_poly(d=n)
        self.test_funcs['poly-100-int-sparse'] = {
            'fun': poly,
            'x0': CArray.ones((n,), dtype=int).tosparse(dtype=int) * 2
        }

    @staticmethod
    def _quadratic_fun(d):
        """Creates a quadratic function in d dimensions."""

        def _quadratic_fun_min(A, b):
            from scipy import linalg
            min_x_scipy = linalg.solve(
                (2 * A).tondarray(), -b.tondarray(), sym_pos=True)
            return CArray(min_x_scipy).ravel()

        A = CArray.eye(d, d)
        b = CArray.ones((d, 1)) * 2

        discr_fun = CFunction.create('quadratic', A, b, c=0)

        min_x = _quadratic_fun_min(A, b)
        min_val = discr_fun.fun(min_x)

        discr_fun.global_min = lambda: min_val
        discr_fun.global_min_x = lambda: min_x

        return discr_fun

    @staticmethod
    def _create_poly(d):
        """Creates a polynomial function in d dimensions."""

        def _poly_fun(x):
            return (x ** 4).sum() + x.sum() ** 2

        def _poly_grad(x):
            return (4 * x ** 3) + 2 * x

        int_fun = CFunction(fun=_poly_fun, gradient=_poly_grad)
        int_fun.global_min = lambda: 0.
        int_fun.global_min_x = lambda: CArray.zeros(d, )

        return int_fun

    def _test_minimize(self, opt_class, fun_id,
                       opt_params=None, minimize_params=None,
                       label=None, out_int=False):
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
        out_int : bool, optional
            If True, output solution should have int dtype. Default False.

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
            raise RuntimeError(
                "mc-cormick always needs the following bounds for "
                "correct optimization: {:}, {:}".format(*fun.bounds())
            )

        min_x = opt.minimize(fun_dict['x0'], **minimize_params)

        self.logger.info("x0: {:}".format(fun_dict['x0']))
        self.logger.info("Found minimum: {:}".format(min_x))
        self.logger.info("Fun value @ minimum: {:}".format(opt.f_opt))

        if fun.global_min_x().size == 2:
            self._plot_optimization(opt, fun_dict['x0'], min_x,
                                    grid_limits=fun_dict['grid_limits'],
                                    method=minimize_params.get('method'),
                                    vmin=fun_dict['vmin'],
                                    vmax=fun_dict['vmax'],
                                    label=label)

        # Round results for easier asserts
        self.assertAlmostEqual(opt.f_opt, fun.global_min(), places=4)
        self.assert_array_almost_equal(
            min_x.todense().ravel(), fun.global_min_x(), decimal=4)

        # Check if the type of the solution is correct
        self.assertEqual(fun_dict['x0'].issparse, min_x.issparse)

        # Check if solution has expected int dtype or not
        self.assertIsSubDtype(min_x.dtype, int if out_int is True else float)

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

        # Plot objective function
        fig.sp.plot_fun(func=CArray.apply_along_axis,
                        plot_background=True,
                        n_grid_points=30, n_colors=25,
                        grid_limits=grid_limits,
                        levels=[0.5], levels_color='gray', levels_style='--',
                        colorbar=True, func_args=(solver.f.fun, 1,),
                        vmin=vmin, vmax=vmax)

        if solver.bounds is not None:  # Plot box constraint
            fig.sp.plot_fun(func=lambda x: solver.bounds.constraint(x),
                            plot_background=False, n_grid_points=20,
                            grid_limits=grid_limits, levels=[0],
                            colorbar=False)

        if solver.constr is not None:  # Plot distance constraint
            fig.sp.plot_fun(func=lambda x: solver.constr.constraint(x),
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
            solver.f.global_min_x().round(2).tolist()))

        test_img_fold_name = 'test_images'
        test_img_fold_path = fm.join(fm.abspath(__file__), test_img_fold_name)
        if not fm.folder_exist(test_img_fold_path):
            fm.make_folder(test_img_fold_path)

        if method is None:
            filename = fm.join(test_img_fold_path,
                               solver.class_type + '-' + solver.f.class_type)
        else:
            filename = fm.join(
                test_img_fold_path,
                solver.class_type + '-' + method + '-' + solver.f.class_type)

        filename += '-' + label if label is not None else ''

        test_img_fold_name = 'test_images'
        if not fm.folder_exist(test_img_fold_name):
            fm.make_folder(test_img_fold_name)

        fig.savefig('{:}.pdf'.format(filename))
