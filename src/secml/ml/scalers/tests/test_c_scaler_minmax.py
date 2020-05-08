from secml.ml.scalers.tests import CScalerTestCases
from sklearn.preprocessing import MinMaxScaler
from secml.array import CArray
from secml.ml.scalers.c_scaler_minmax import CScalerMinMax


class TestCScalerMinMax(CScalerTestCases):
    """Unittests for CScalerMinMax."""

    def _compare_scalers(self, scaler, scaler_sklearn,
                         array, convert_to_dense=False):
        """Compare wrapped scikit-learn scaler to the unwrapped scaler.

        Parameters
        ----------
        array : CArray
        scaler : A wrapped CScaler
        scaler_sklearn
            Scikit-learn normalizer (from `sklearn.preprocessing`).
        convert_to_dense : bool, optional
            If True the data used by the SkLearn scaler will be converted to
            dense.

        Returns
        -------
        scaler_sklearn
            Trained Scikit-learn normalizer (from `sklearn.preprocessing`).
        scaler : CScaler
            Trained normalizer.

        """
        scaler, scaler_sklearn = \
            super(TestCScalerMinMax, self)._compare_scalers(scaler,
                                                            scaler_sklearn,
                                                            array,
                                                            convert_to_dense)

        self.logger.info("Testing out of range normalization")

        array_sk = array.get_data() if convert_to_dense is False \
            else array.tondarray()

        # Sklearn normalizer (requires float dtype input)
        transform_sklearn = CArray(scaler_sklearn.transform(array_sk * 2))

        # Our normalizer
        transform = scaler.forward(array * 2)

        self.logger.info("Correct result is:\n{:}".format(transform_sklearn))
        self.logger.info("Our result is:\n{:}".format(transform))

        self.assert_array_almost_equal(transform_sklearn, transform)

        return scaler, scaler_sklearn

    def test_forward(self):
        """Test for `.forward()` method."""
        self._compare_scalers(CScalerMinMax(), MinMaxScaler(),
                              self.array_dense)
        self._compare_scalers(CScalerMinMax(), MinMaxScaler(),
                              self.row_dense.atleast_2d())
        self._compare_scalers(CScalerMinMax(), MinMaxScaler(),
                              self.column_dense)

    def test_chain(self):
        """Test a chain of preprocessors."""
        self._test_chain(self.array_dense,
                         ['minmax', 'pca', 'minmax'],
                         [{'feature_range': (-5, 5)}, {},
                          {'feature_range': (0, 1)}])

    def test_chain_gradient(self):
        """Check gradient of a chain of preprocessors."""
        self._test_chain_gradient(self.array_dense,
                                  ['minmax', 'std', 'minmax'],
                                  [{'feature_range': (-5, 5)}, {},
                                   {'feature_range': (0, 1)}])


if __name__ == '__main__':
    CScalerTestCases.main()
