from secml.testing import CUnitTest
from secml.ml.tests import CModuleTestCases


class CPreProcessTestCases(CModuleTestCases):
    """Unittests interface for CPreProcess."""

    # TODO: this class extends CModuleTestCases with test on inverse_transform
    #  consider refactoring to avoid code duplication
    def _test_chain(self, x, class_type_list, kwargs_list, y=None):
        """Tests if preprocess chain and manual chaining yield same result."""
        chain, pre_list = self._create_chain(class_type_list, kwargs_list)

        chain = chain.fit(x, y=y)
        self.logger.info("Preprocessors chain:\n{:}".format(chain))

        x_chain = chain.forward(x)
        self.logger.info("Trasformed X (chain):\n{:}".format(x_chain))

        # Train the manual chain and transform
        x_manual = x
        for pre in pre_list:
            x_manual = pre.fit_forward(x_manual, y=y)

        self.logger.info("Trasformed X (manual):\n{:}".format(x_manual))
        self.assert_allclose(x_chain, x_manual)

        # Reverting array (if available)
        try:
            x_chain_revert = chain.inverse_transform(x_chain)
            self.logger.info("Reverted X (chain):\n{:}".format(x_chain_revert))
            self.logger.info("Original X:\n{:}".format(x))
            self.assert_array_almost_equal(x_chain_revert, x)
        except NotImplementedError:
            self.logger.info("inverse_transform not available")

        return x_chain


if __name__ == '__main__':
    CUnitTest.main()
