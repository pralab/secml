from secml.optim.optimizers.tests import COptimizerTestCases

from secml.optim.optimizers import COptimizerGradBLS
from secml.optim.constraints import CConstraintBox, CConstraintL2


class TestCOptimizerGradBLS(COptimizerTestCases):
    """Unittests for COptimizerGradBLS."""

    def test_minimize_3h_camel(self):
        """Test for COptimizer.minimize() method on 3h-camel fun."""
        opt_params = {'eta': 1e-6, 'eta_min': 1e-4, 'eps': 1e-12}

        self._test_minimize(
            COptimizerGradBLS, '3h-camel', opt_params=opt_params)

        # Test discrete optimization with float eta
        opt_params = {'eta': 0.5, 'eta_min': 0.5, 'eps': 1e-12,
                      'discrete': True,
                      'bounds': CConstraintBox(lb=-1, ub=1)}

        self._test_minimize(COptimizerGradBLS, '3h-camel',
                            opt_params=opt_params,
                            label='discrete')

    def test_minimize_beale(self):
        """Test for COptimizer.minimize() method on beale fun."""
        opt_params = {'eta': 1e-6, 'eta_min': 1e-4, 'eps': 1e-12}

        self._test_minimize(
            COptimizerGradBLS, 'beale', opt_params=opt_params)

        # Test discrete optimization with float eta
        opt_params = {'eta': 1e-6, 'eta_min': 1e-4, 'eps': 1e-12,
                      'discrete': True,
                      'bounds': CConstraintBox(lb=0, ub=4)}

        self._test_minimize(COptimizerGradBLS, 'beale',
                            opt_params=opt_params,
                            label='discrete')

    def test_minimize_mc_cormick(self):
        """Test for COptimizer.minimize() method on mc-cormick fun."""
        opt_params = {'eta': 1e-6, 'eta_min': 1e-4, 'eps': 1e-12}

        self._test_minimize(
            COptimizerGradBLS, 'mc-cormick', opt_params=opt_params)

    def test_minimize_rosenbrock(self):
        """Test for COptimizer.minimize() method on rosenbrock fun."""
        opt_params = {'eta': 1e-6, 'eta_min': 1e-6, 'eps': 1e-12}

        self._test_minimize(
            COptimizerGradBLS, 'rosenbrock', opt_params=opt_params)

    def test_minimize_discrete(self):
        """Test for COptimizer.minimize() method in discrete space."""

        # test a simple function without any bound
        opt_params = {'eta': 1, 'eta_min': 1, 'eps': 1e-12,
                      'discrete': True}

        # both the starting point and eta are integer,
        # therefore we expect an integer solution
        self._test_minimize(COptimizerGradBLS, 'quad-2',
                            opt_params=opt_params,
                            label='quad-2-discrete',
                            out_int=True)

        # Testing bounded optimization
        opt_params = {'eta': 1, 'eta_min': 1, 'eps': 1e-12,
                      'discrete': True,
                      'bounds': CConstraintBox(lb=-2, ub=3)}

        self._test_minimize(
            COptimizerGradBLS, 'quad-2',
            opt_params=opt_params,
            label='quad-2-discrete-bounded',
            out_int=True)

        self._test_minimize(
            COptimizerGradBLS, 'quad-100-sparse',
            opt_params=opt_params,
            label='quad-100-sparse-discrete-bounded',
            out_int=True)

        # Testing bounded optimization
        opt_params = {'eta': 1, 'eta_min': 1, 'eps': 1e-12,
                      'discrete': True,
                      'bounds': CConstraintBox(lb=-1, ub=1)}

        self._test_minimize(
            COptimizerGradBLS, 'exp-sum-2',
            opt_params=opt_params,
            label='exp-sum-discrete-bounded',
        )

        self._test_minimize(
            COptimizerGradBLS, 'exp-sum-100-int',
            opt_params=opt_params,
            label='exp-sum-int-discrete-bounded',
            out_int=True)

        self._test_minimize(
            COptimizerGradBLS, 'exp-sum-100-int-sparse',
            opt_params=opt_params,
            label='exp-sum-int-sparse-discrete-bounded',
            out_int=True)

        # Discrete optimization + L2 constraint is not supported
        with self.assertRaises(NotImplementedError):
            opt_params = {'eta': 1e-6, 'eta_min': 1e-4, 'eps': 1e-12,
                          'discrete': True, 'constr': CConstraintL2()}
            self._test_minimize(
                COptimizerGradBLS, 'beale', opt_params=opt_params)


if __name__ == '__main__':
    COptimizerTestCases.main()
