from secml.testing import CUnitTest
from secml.array import CArray
from secml.ml.tests import CModuleTestCases


class CScalerTestCases(CModuleTestCases):
    """Unittests interface for Normalizers."""

    def _compare_scalers(self, scaler, scaler_sklearn,
                         array, convert_to_dense=False):
        """Compare wrapped scikit-learn scaler to the unwrapped scaler.

        Parameters
        ----------
        array : CArray
        scaler : A wrapped CScaler
        scaler_sklearn
            Scikit-learn normalizer.
        convert_to_dense : bool, optional
            If True the data used by the SkLearn scaler will be converted

            to dense.

        Returns
        -------
        scaler_sklearn
            Trained Scikit-learn normalizer (from `sklearn.preprocessing`).
        scaler : CScaler
            Trained normalizer.

        """
        self.logger.info("Original array is:\n{:}".format(array))

        array_sk = array.get_data() if convert_to_dense is False \
            else array.tondarray()

        # Sklearn normalizer
        scaler_sklearn.fit(array_sk, None)
        transform_sklearn = CArray(scaler_sklearn.transform(array_sk))

        # Our normalizer
        scaler._fit(array)
        transform = scaler.forward(array)

        self.logger.info("sklearn result is:\n{:}".format(transform_sklearn))
        self.logger.info("Our result is:\n{:}".format(transform))

        self.assert_array_almost_equal(transform_sklearn, transform)

        return scaler, scaler_sklearn

    def _test_chain(self, x, class_type_list, kwargs_list, y=None):
        """Tests if preprocess chain and manual chaining yield same result."""
        x_chain = super(CScalerTestCases, self)._test_chain(
            x, class_type_list, kwargs_list, y)

        self.assertEqual((self.array_dense.shape[0],
                          self.array_dense.shape[1] - 1), x_chain.shape)

        return x_chain

    def _test_chain_gradient(self, x, class_type_list, kwargs_list, y=None):
        """Tests if gradient preprocess chain and
        gradient of manual chaining yield same result."""
        grad_chain = super(CScalerTestCases, self)._test_chain_gradient(
            x, class_type_list, kwargs_list, y)

        self.assertEqual((self.array_dense.shape[1],), grad_chain.shape)

        return grad_chain


if __name__ == '__main__':
    CUnitTest.main()
