from secml.ml.scalers.tests import CScalerTestCases

from sklearn.preprocessing import StandardScaler

from secml.ml.scalers import CScalerStd


class TestCScalerStd(CScalerTestCases):
    """Unittests for CScalerStd."""

    def test_forward(self):
        """Test for `.forward()` method."""
        # mean should not be used for sparse arrays
        for with_std in (True, False):
            self.logger.info("Testing using std? {:}".format(with_std))

            self._compare_scalers(CScalerStd(with_std=with_std),
                                  StandardScaler(with_std=with_std),
                                  self.array_dense)
            self._compare_scalers(CScalerStd(with_std=with_std,
                                             with_mean=False),
                                  StandardScaler(with_std=with_std,
                                                 with_mean=False),
                                  self.array_sparse)
            self._compare_scalers(CScalerStd(with_std=with_std),
                                  StandardScaler(with_std=with_std),
                                  self.row_dense.atleast_2d())
            self._compare_scalers(CScalerStd(with_std=with_std,
                                             with_mean=False),
                                  StandardScaler(with_std=with_std,
                                                 with_mean=False),
                                  self.row_sparse)
            self._compare_scalers(CScalerStd(with_std=with_std),
                                  StandardScaler(with_std=with_std),
                                  self.column_dense)
            self._compare_scalers(CScalerStd(with_std=with_std,
                                             with_mean=False),
                                  StandardScaler(with_std=with_std,
                                                 with_mean=False),
                                  self.column_sparse)

    def test_mean_std(self):
        """Test using specific mean/std."""
        for (mean, std) in [(1.5, 0.1),
                            ((1.0, 1.1, 1.2, 1.3), (0.0, 0.1, 0.2, 0.3))]:
            for array in [self.array_dense, self.array_sparse]:
                self.logger.info("Original array is:\n{:}".format(array))
                self.logger.info(
                    "Normalizing using mean: {:} std: {:}".format(mean, std))

                n = CScalerStd(with_mean=not array.issparse)

                n._fit(array)
                n.sklearn_scaler.mean = mean
                n.sklearn_scaler.std = std

                out = n._forward(array)

                self.logger.info("Result is:\n{:}".format(out))

                out_mean = out.mean(axis=0, keepdims=False)
                out_std = out.std(axis=0, keepdims=False)

                self.logger.info("Result mean is:\n{:}".format(out_mean))
                self.logger.info("Result std is:\n{:}".format(out_std))

    # def _array_test(array, ):

    def test_chain(self):
        """Test a chain of preprocessors."""
        self._test_chain(self.array_dense,
                         ['minmax', 'pca', 'std'],
                         [{'feature_range': (-5, 5)}, {}, {}])

    def test_chain_gradient(self):
        """Check gradient of a chain of preprocessors."""
        self._test_chain_gradient(self.array_dense,
                                  ['minmax', 'std'],
                                  [{'feature_range': (-5, 5)}, {}])


if __name__ == '__main__':
    CScalerTestCases.main()
