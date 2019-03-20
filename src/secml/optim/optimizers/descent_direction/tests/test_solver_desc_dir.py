from secml.optim.optimizers.tests import COptimizerTestCases

from secml.optim.optimizers import CSolverDescDir
from secml.optim.constraints import CConstraintBox


class TestOptimizerDescDir(COptimizerTestCases):
    """Unittests for COptimizerDescDir."""

    # FIXME: STILL BUGGED. SEE #20
    # def test_minimize_3h_camel(self):
    #     """Test for COptimizer.minimize() method on 3h-camel fun."""
    #     opt_params = {'eta': 1e-6, 'eta_min': 1e-4, 'eps': 1e-12,
    #                   'bounds': CConstraintBox()}  # FIXME: FIX #372 AND REMOVE BOUNDS
    #
    #     self._test_minimize(
    #         CSolverDescDir, '3h-camel', opt_params=opt_params)

    # FIXME: DOES NOT REACH THE GLOBAL MIN (PARAMS PROBLEM?)
    # def test_minimize_beale(self):
    #     """Test for COptimizer.minimize() method on beale fun."""
    #     opt_params = {'eta': 1e-6, 'eta_min': 1e-4, 'eps': 1e-12,
    #                   'bounds': CConstraintBox()}  # FIXME: FIX #372 AND REMOVE BOUNDS
    #
    #     self._test_minimize(
    #         CSolverDescDir, 'beale', opt_params=opt_params)

    # FIXME: DOES NOT REACH THE GLOBAL MIN (PARAMS PROBLEM?)
    # def test_minimize_mc_cormick(self):
    #     """Test for COptimizer.minimize() method on mc-cormick fun."""
    #     opt_params = {'eta': 1e-6, 'eta_min': 1e-4, 'eps': 1e-12,
    #                   'bounds': CConstraintBox()}  # FIXME: FIX #372 AND REMOVE BOUNDS
    #
    #     self._test_minimize(
    #         CSolverDescDir, 'mc-cormick', opt_params=opt_params)

    # FIXME: DOES NOT REACH THE GLOBAL MIN (PARAMS PROBLEM?)
    # def test_minimize_rosenbrock(self):
    #     """Test for COptimizer.minimize() method on rosenbrock fun."""
    #     opt_params = {'eta': 1e-6, 'eta_min': 1e-4, 'eps': 1e-12,
    #                   'bounds': CConstraintBox()}  # FIXME: FIX #372 AND REMOVE BOUNDS
    #
    #     self._test_minimize(
    #         CSolverDescDir, 'rosenbrock', opt_params=opt_params)

    # TODO: IMPROVE THIS TEST (DATA IS NOT SPARSE)
    def test_minimize_discrete(self):
        """Test for COptimizer.minimize() method in discrete space."""

        opt_params = {'eta': 1, 'eta_min': 1, 'eps': 1,
                      'discrete': True,
                      'bounds': CConstraintBox(lb=-2, ub=3)}

        self._test_minimize(CSolverDescDir, 'quadratic', opt_params=opt_params)


if __name__ == '__main__':
    COptimizerTestCases.main()

