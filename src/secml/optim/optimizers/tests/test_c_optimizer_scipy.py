from secml.optim.optimizers.tests import COptimizerTestCases

from secml.optim.optimizers import COptimizerScipy


class TestCOptimizerScipy(COptimizerTestCases):
    """Unittests for COptimizerScipy."""

    def test_minimize(self):
        """Test for COptimizer.minimize() method."""

        minimize_params = {'method': 'L-BFGS-B', 'options': {'gtol': 1e-6}}

        self._test_minimize(
            COptimizerScipy, '3h-camel', minimize_params=minimize_params)
        self._test_minimize(
            COptimizerScipy, 'beale', minimize_params=minimize_params)
        self._test_minimize(
            COptimizerScipy, 'mc-cormick', minimize_params=minimize_params)
        self._test_minimize(
            COptimizerScipy, 'rosenbrock', minimize_params=minimize_params)


if __name__ == "__main__":
    COptimizerTestCases.main()
