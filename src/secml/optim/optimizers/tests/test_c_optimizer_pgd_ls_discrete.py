from secml.optim.optimizers.tests import COptimizerTestCases

from secml.optim.optimizers import COptimizerPGDLS
from secml.optim.constraints import CConstraintBox, CConstraintL2


class TestCOptimizerPGDLSDiscrete(COptimizerTestCases):
    """Unittests for COptimizerPGDLSDiscrete."""

    def test_minimize_3h_camel(self):
        """Test for COptimizer.minimize() method on 3h-camel fun. This
        function tests the optimization in discrete space, with a floating
        eta and an integer starting point. The solution expected by this
        test is a float vector."""

        opt_params = {'eta': 0.5, 'eta_min': 0.5, 'eps': 1e-12,
                      'discrete': True,
                      'bounds': CConstraintBox(lb=-1, ub=1)}

        self._test_minimize(COptimizerPGDLS, '3h-camel',
                            opt_params=opt_params,
                            label='discrete')

    def test_minimize_beale(self):
        """Test for COptimizer.minimize() method on 3h-camel fun. This
        function tests the optimization in discrete space, with a floating
        eta and an integer starting point. The solution expected by this
        test is a float vector."""
        opt_params = {'eta': 1e-6, 'eta_min': 1e-4, 'eps': 1e-12,
                      'discrete': True,
                      'bounds': CConstraintBox(lb=0, ub=4)}

        self._test_minimize(COptimizerPGDLS, 'beale',
                            opt_params=opt_params,
                            label='discrete')

    def test_minimize_quad2d_no_bound(self):
        """Test for COptimizer.minimize() method on a quadratic function in
        a 2-dimensional space. This function tests the optimization in discrete
        space, with an integer eta, an integer starting point and without any
        bound. The solution expected by this test is an integer vector."""

        # test a simple function without any bound
        opt_params = {'eta': 1, 'eta_min': 1, 'eps': 1e-12,
                      'discrete': True}

        # both the starting point and eta are integer,
        # therefore we expect an integer solution
        self._test_minimize(COptimizerPGDLS, 'quad-2',
                            opt_params=opt_params,
                            label='quad-2-discrete',
                            out_int=True)

    def test_minimize_quad2d_bound(self):
        """Test for COptimizer.minimize() method on a quadratic function in
        a 2-dimensional space. This function tests the optimization in discrete
        space, with an integer eta, an integer starting point and with a box
        constraint. The solution expected by this test is an integer vector."""

        # Testing bounded optimization
        opt_params = {'eta': 1, 'eta_min': 1, 'eps': 1e-12,
                      'discrete': True,
                      'bounds': CConstraintBox(lb=-2, ub=3)}

        self._test_minimize(
            COptimizerPGDLS, 'quad-2',
            opt_params=opt_params,
            label='quad-2-discrete-bounded',
            out_int=True)

    def test_minimize_quad100d_sparse(self):
        """Test for COptimizer.minimize() method on a quadratic function in
        a 100-dimensional space. This function tests the optimization in
        discrete space, with an integer eta, an integer and sparse starting
        point with box constraint. The solution expected by this test is an
        integer and sparse vector."""

        opt_params = {'eta': 1, 'eta_min': 1, 'eps': 1e-12,
                      'discrete': True,
                      'bounds': CConstraintBox(lb=-2, ub=3)}

        self._test_minimize(
            COptimizerPGDLS, 'quad-100-sparse',
            opt_params=opt_params,
            label='quad-100-sparse-discrete-bounded',
            out_int=True)

    def test_minimize_expsum2d_bounded(self):
        """Test for COptimizer.minimize() method on a polynomial function in
        a 2-dimensional space. This function tests the optimization in
        discrete space, with an integer eta, an floating starting point with
        a box constraint. The solution expected by this test is an integer
        vector."""

        opt_params = {'eta': 1, 'eta_min': 1, 'eps': 1e-12,
                      'discrete': True,
                      'bounds': CConstraintBox(lb=-1, ub=1)}

        self._test_minimize(
            COptimizerPGDLS, 'poly-2',
            opt_params=opt_params,
            label='poly-discrete-bounded',
        )

    def test_minimize_expsum100d_bounded(self):
        """Test for COptimizer.minimize() method on a polynomial function in
        a 2-dimensional space. This function tests the optimization in
        discrete space, with an integer eta, an integer starting point with
        a box constraint. The solution of this problem is integer and sparse
        (it is a zero vector)"""
        opt_params = {'eta': 1, 'eta_min': 1, 'eps': 1e-12,
                      'discrete': True,
                      'bounds': CConstraintBox(lb=-1, ub=1)}

        self._test_minimize(
            COptimizerPGDLS, 'poly-100-int',
            opt_params=opt_params,
            label='poly-int-discrete-bounded',
            out_int=True)

    def test_minimize_poly_100d_bounded_sparse(self):
        """Test for COptimizer.minimize() method on a polynomial function in
        a 100-dimensional space. This function tests the optimization in
        discrete space, with an integer eta, an integer and sparse starting
        point (it is a zero vector) with a box constraint. The solution
        expected by this test is an integer and sparse vector."""
        opt_params = {'eta': 1, 'eta_min': 1, 'eps': 1e-12,
                      'discrete': True,
                      'bounds': CConstraintBox(lb=-1, ub=1)}

        self._test_minimize(
            COptimizerPGDLS, 'poly-100-int-sparse',
            opt_params=opt_params,
            label='poly-int-sparse-discrete-bounded',
            out_int=True)

    def test_minimize_poly_100d_lc_constr(self):
        """Discrete optimization + L2 constraint is not supported. This
        function tests if the optimzier raises correctly an error when it
        receives as parameter discrete = True and an L2 constraint."""

        with self.assertRaises(NotImplementedError):
            opt_params = {'eta': 1e-6, 'eta_min': 1e-4, 'eps': 1e-12,
                          'discrete': True, 'constr': CConstraintL2()}
            self._test_minimize(
                COptimizerPGDLS, 'beale', opt_params=opt_params)


if __name__ == '__main__':
    COptimizerTestCases.main()
