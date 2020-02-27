from secml.ml.features.tests import CPreProcessTestCases

from sklearn.preprocessing import Normalizer

from secml.array import CArray
from secml.ml.features.normalization import CNormalizerUnitNorm


class TestCNormalizerUnitNorm(CPreProcessTestCases):
    """Unittest for CNormalizerUnitNorm."""

    def test_norm_unitnorm(self):
        """Test for CNormalizerUnitNorm."""

        def sklearn_comp(array):

            self.logger.info("Original array is:\n{:}".format(array))

            # Sklearn normalizer (requires float dtype input)
            target = CArray(Normalizer().fit_transform(
                array.astype(float).get_data()))
            # Our normalizer
            result = CNormalizerUnitNorm().fit_transform(array)

            self.logger.info("Correct result is:\n{:}".format(target))
            self.logger.info("Our result is:\n{:}".format(result))

            self.assert_array_almost_equal(target, result)

        sklearn_comp(self.array_dense)
        sklearn_comp(self.array_sparse)
        sklearn_comp(self.row_dense.atleast_2d())
        sklearn_comp(self.row_sparse)
        sklearn_comp(self.column_dense)
        sklearn_comp(self.column_sparse)

    def test_chain(self):
        """Test a chain of preprocessors."""
        x_chain = self._test_chain(
            self.array_dense,
            ['min-max', 'pca', 'unit-norm'],
            [{'feature_range': (-5, 5)}, {}, {}]
        )

        # Expected shape is (3, 3), as pca max n_components is 4-1
        self.assertEqual((self.array_dense.shape[0],
                          self.array_dense.shape[1]-1), x_chain.shape)


if __name__ == '__main__':
    CPreProcessTestCases.main()
