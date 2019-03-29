from secml.optim.optimizers.tests import COptimizerTestCases
from secml.optim.optimizers import COptimizerScipy
from secml.optim.constraints import CConstraintBox

TEST_FUNCS = ['3h-camel', 'beale', 'mc-cormick', 'rosenbrock']


class TestCOptimizerScipy(COptimizerTestCases):
    """Unittests for COptimizerScipy."""

    def test_minimize(self):
        """Test for COptimizer.minimize() method."""

        params = {'method': None, 'options': {'gtol': 1e-6}}

        for fun in TEST_FUNCS:
            # test using BFGS scipy solver
            params['method'] = 'BFGS'
            self._test_minimize(
                COptimizerScipy, fun, opt_params={},
                minimize_params=params)

            # test using L-BFGS-B scipy solver (supports bounds)
            params['method'] = 'L-BFGS-B'
            bounds = CConstraintBox(lb=-2, ub=3)  # fake box
            self._test_minimize(
                COptimizerScipy, fun, opt_params={'bounds': bounds},
                minimize_params=params)


if __name__ == "__main__":
    COptimizerTestCases.main()
