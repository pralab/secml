from secml.optim.optimizers.tests import COptimizerTestCases

from secml.optim.optimizers import COptimizerGradBLS


class TestCOptimizerGradBLS(COptimizerTestCases):
    """Unittests for COptimizerGradBLS."""

    def test_minimize_3h_camel(self):
        """Test for COptimizer.minimize() method on 3h-camel fun."""
        opt_params = {'eta': 1e-6, 'eta_min': 1e-4, 'eps': 1e-12}

        self._test_minimize(
            COptimizerGradBLS, '3h-camel', opt_params=opt_params)

    def test_minimize_beale(self):
        """Test for COptimizer.minimize() method on beale fun."""
        opt_params = {'eta': 1e-6, 'eta_min': 1e-4, 'eps': 1e-12}

        self._test_minimize(
            COptimizerGradBLS, 'beale', opt_params=opt_params)

    def test_minimize_mc_cormick(self):
        """Test for COptimizer.minimize() method on mc-cormick fun."""
        from secml.optim.function import CFunctionMcCormick
        from secml.optim.constraints import CConstraintBox
        opt_params = {'eta': 1e-6, 'eta_min': 1e-4, 'eps': 1e-12,
                      'bounds': CConstraintBox(*CFunctionMcCormick.bounds())}

        self._test_minimize(
            COptimizerGradBLS, 'mc-cormick', opt_params=opt_params)

    def test_minimize_rosenbrock(self):
        """Test for COptimizer.minimize() method on rosenbrock fun."""
        opt_params = {'eta': 1e-6, 'eta_min': 1e-6, 'eps': 1e-12}

        self._test_minimize(
            COptimizerGradBLS, 'rosenbrock', opt_params=opt_params)


if __name__ == '__main__':
    COptimizerTestCases.main()
