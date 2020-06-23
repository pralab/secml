from secml.ml.scalers.tests import CScalerTestCases

from sklearn.preprocessing import Normalizer

from secml.ml.scalers import CScalerNorm


class TestCScalerNorm(CScalerTestCases):
    """Unittests for CScalerNorm."""

    def test_forward(self):
        """Test for `.forward()` method."""
        for norm_type in ["l1", "l2", "max"]:
            self._compare_scalers(CScalerNorm(norm=norm_type),
                                  Normalizer(norm=norm_type),
                                  self.array_dense)
            self._compare_scalers(CScalerNorm(norm=norm_type),
                                  Normalizer(norm=norm_type),
                                  self.array_sparse)
            self._compare_scalers(CScalerNorm(norm=norm_type),
                                  Normalizer(norm=norm_type),
                                  self.row_dense.atleast_2d())
            self._compare_scalers(CScalerNorm(norm=norm_type),
                                  Normalizer(norm=norm_type),
                                  self.row_sparse)
            self._compare_scalers(CScalerNorm(norm=norm_type),
                                  Normalizer(norm=norm_type),
                                  self.column_dense)
            self._compare_scalers(CScalerNorm(norm=norm_type),
                                  Normalizer(norm=norm_type),
                                  self.column_sparse)

    def test_chain(self):
        """Test a chain of preprocessors."""
        self._test_chain(self.array_dense,
                         ['minmax', 'pca', 'norm'],
                         [{'feature_range': (-5, 5)}, {}, {}])

    def test_chain_gradient(self):
        """Check gradient of a chain of preprocessors."""
        self._test_chain_gradient(self.array_dense,
                                  ['minmax', 'std', 'norm'],
                                  [{'feature_range': (-5, 5)}, {}, {}])


if __name__ == '__main__':
    CScalerTestCases.main()
