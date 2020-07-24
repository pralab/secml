from secml.ml.features.normalization.tests import CNormalizerTestCases

from sklearn.preprocessing import MinMaxScaler

from secml.array import CArray
from secml.ml.features.normalization import CNormalizerMinMax


class TestCNormalizerMinMax(CNormalizerTestCases):
    """Unittests for CNormalizerMinMax."""

    def _sklearn_comp(self, array, norm_sklearn, norm, sparse=False):
        """Compare scikit-learn normalizer with our implementation.

        Parameters
        ----------
        array : CArray
        norm_sklearn
            Scikit-learn normalizer (from `sklearn.preprocessing`).
        norm : CNormalizer
        sparse : bool, optional
            If False (default) sklearn normalizer only supports dense data.

        Returns
        -------
        norm_sklearn
            Trained Scikit-learn normalizer (from `sklearn.preprocessing`).
        norm : CNormalizer
            Trained normalizer.

        """
        norm_sklearn, norm = \
            super(TestCNormalizerMinMax, self)._sklearn_comp(
                array, norm_sklearn, norm, sparse)

        self.logger.info("Testing out of range normalization")

        array_sk = array.get_data() if sparse is True else array.tondarray()

        # Sklearn normalizer (requires float dtype input)
        transform_sklearn = CArray(norm_sklearn.transform(array_sk * 2))

        # Our normalizer
        transform = norm.transform(array * 2)

        self.logger.info("Correct result is:\n{:}".format(transform_sklearn))
        self.logger.info("Our result is:\n{:}".format(transform))

        self.assert_array_almost_equal(transform_sklearn, transform)

        return norm_sklearn, norm

    def test_transform(self):
        """Test for `.transform()` method."""
        self._sklearn_comp(
            self.array_dense, MinMaxScaler(), CNormalizerMinMax())
        self._sklearn_comp(
            self.array_sparse, MinMaxScaler(), CNormalizerMinMax())
        self._sklearn_comp(
            self.row_dense.atleast_2d(), MinMaxScaler(), CNormalizerMinMax())
        self._sklearn_comp(
            self.row_sparse, MinMaxScaler(), CNormalizerMinMax())
        self._sklearn_comp(
            self.column_dense, MinMaxScaler(), CNormalizerMinMax())
        self._sklearn_comp(
            self.column_sparse, MinMaxScaler(), CNormalizerMinMax())

    def test_chain(self):
        """Test a chain of preprocessors."""
        self._test_chain(self.array_dense,
                         ['min-max', 'pca', 'min-max'],
                         [{'feature_range': (-5, 5)}, {},
                          {'feature_range': (0, 1)}])

    def test_chain_gradient(self):
        """Check gradient of a chain of preprocessors."""
        self._test_chain_gradient(self.array_dense,
                                  ['min-max', 'mean-std', 'min-max'],
                                  [{'feature_range': (-5, 5)}, {},
                                   {'feature_range': (0, 1)}])


if __name__ == '__main__':
    CNormalizerTestCases.main()
