from secml.optim.optimizers.tests import COptimizerTestCases
from secml.optim.optimizers import COptimizerScipy
from secml.optim.constraints import CConstraintBox

TEST_FUNCS = ['3h-camel', 'beale', 'mc-cormick', 'rosenbrock']


class TestCOptimizerScipy(COptimizerTestCases):
    """Unittests for COptimizerScipy."""

    def test_minimize_3h_camel(self):
        """Test for COptimizer.minimize() method on 3h-camel fun."""

        self._test_minimize(
            COptimizerScipy, '3h-camel', opt_params={},
            minimize_params={
                'method': 'BFGS', 'options': {'gtol': 1e-6}})

        # test using L-BFGS-B scipy solver (supports bounds)
        bounds = CConstraintBox(lb=-2, ub=3)  # fake box
        self._test_minimize(
            COptimizerScipy, '3h-camel', opt_params={'bounds': bounds},
            minimize_params={
                'method': 'L-BFGS-B', 'options': {'gtol': 1e-6}})

    def test_minimize_beale(self):
        """Test for COptimizer.minimize() method on beale fun."""

        self._test_minimize(
            COptimizerScipy, 'beale', opt_params={},
            minimize_params={
                'method': 'BFGS', 'options': {'gtol': 1e-6}})

        # test using L-BFGS-B scipy solver (supports bounds)
        bounds = CConstraintBox(lb=-2, ub=3)  # fake box
        self._test_minimize(
            COptimizerScipy, 'beale', opt_params={'bounds': bounds},
            minimize_params={
                'method': 'L-BFGS-B', 'options': {'gtol': 1e-6}})

    def test_minimize_mc_cormick(self):
        """Test for COptimizer.minimize() method on mc-cormick fun."""
        # mc-cormick always needs the bounds
        # test using L-BFGS-B scipy solver (supports bounds)
        bounds = CConstraintBox(lb=-2, ub=3)  # fake box
        self._test_minimize(
            COptimizerScipy, 'mc-cormick', opt_params={'bounds': bounds},
            minimize_params={
                'method': 'L-BFGS-B', 'options': {'gtol': 1e-6}})

    def test_minimize_rosenbrock(self):
        """Test for COptimizer.minimize() method on rosenbrock fun."""

        self._test_minimize(
            COptimizerScipy, 'rosenbrock', opt_params={},
            minimize_params={
                'method': 'BFGS', 'options': {'gtol': 1e-6}})

        # test using L-BFGS-B scipy solver (supports bounds)
        bounds = CConstraintBox(lb=-2, ub=3)  # fake box
        self._test_minimize(
            COptimizerScipy, 'rosenbrock', opt_params={'bounds': bounds},
            minimize_params={
                'method': 'L-BFGS-B', 'options': {'gtol': 1e-6}})


if __name__ == "__main__":
    COptimizerTestCases.main()
