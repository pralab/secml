from secml.ml.features.normalization.tests import CNormalizerTestCases
from sklearn.preprocessing import MinMaxScaler
from secml.ml.features.normalization import CNormalizerMinMax
from secml.array import CArray


class TestCNormalizerMinMax(CNormalizerTestCases):
    """Unittest for CNormalizerMinMax."""

    def _sklearn_comp(self, array, norm_sklearn, norm, sparse=False):
        """Tests if the sklearn normalizer (MinMaxScaler) and
        our normalizer (CNormalizerMinMax) yield same result.
        """
        target, result, sk_norm, our_norm, array_sk = super(
            TestCNormalizerMinMax, self)._sklearn_comp(
            array, norm_sklearn, norm, sparse)

        # Testing out of range normalization

        self.logger.info("Testing out of range normalization")

        # Sklearn normalizer (requires float dtype input)
        target = CArray(sk_norm.transform(array_sk * 2))

        # Our normalizer
        result = our_norm.transform(array * 2)

        self.logger.info("Correct result is:\n{:}".format(target))
        self.logger.info("Our result is:\n{:}".format(result))

        self.assert_array_almost_equal(target, result)

        return target, result, sk_norm, our_norm, array_sk

    def test_norm_minmax(self):
        """Test for CNormalizerMinMax."""
        self._sklearn_comp(self.array_dense, MinMaxScaler(),
                           CNormalizerMinMax())
        self._sklearn_comp(self.array_sparse, MinMaxScaler(),
                           CNormalizerMinMax())
        self._sklearn_comp(self.row_dense.atleast_2d(), MinMaxScaler(),
                           CNormalizerMinMax())
        self._sklearn_comp(self.row_sparse, MinMaxScaler(),
                           CNormalizerMinMax())
        self._sklearn_comp(self.column_dense, MinMaxScaler(),
                           CNormalizerMinMax())
        self._sklearn_comp(self.column_sparse, MinMaxScaler(),
                           CNormalizerMinMax())

    def test_chain(self):
        """Tests a chain of preprocessors related to CNormalizerMinMax."""
        self._test_chain(self.array_dense,
                         ['min-max', 'pca', 'min-max'],
                         [{'feature_range': (-5, 5)}, {},
                          {'feature_range': (0, 1)}])
        # Expected shape is (3, 3), as pca max n_components is 4-1

    def test_chain_gradient(self):
        """Tests the gradient of a chain of preprocessors
        related to CNormalizerMinMax.
        """
        # Expected shape is (n_feats, ), so (4, )
        self._test_chain_gradient(self.array_dense,
                                  ['min-max', 'mean-std', 'min-max'],
                                  [{'feature_range': (-5, 5)}, {},
                                   {'feature_range': (0, 1)}])


if __name__ == '__main__':
    CNormalizerTestCases.main()
