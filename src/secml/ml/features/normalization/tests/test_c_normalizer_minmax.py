from secml.ml.features.normalization.tests import CNormalizerTestCases
from sklearn.preprocessing import MinMaxScaler
from secml.ml.features.normalization import CNormalizerMinMax


class TestCNormalizerMinMax(CNormalizerTestCases):
    """Unittest for CNormalizerMinMax."""
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
        self._sklearn_comp(self.array_dense*2, MinMaxScaler(),
                           CNormalizerMinMax())
        self._sklearn_comp(self.array_sparse*2, MinMaxScaler(),
                           CNormalizerMinMax())
        self._sklearn_comp(self.row_dense.atleast_2d()*2, MinMaxScaler(),
                           CNormalizerMinMax())
        self._sklearn_comp(self.row_sparse*2, MinMaxScaler(),
                           CNormalizerMinMax())
        self._sklearn_comp(self.column_dense*2, MinMaxScaler(),
                           CNormalizerMinMax())
        self._sklearn_comp(self.column_sparse*2, MinMaxScaler(),
                           CNormalizerMinMax())

    def test_chain(self):
        """Test a chain of preprocessors."""
        self._test_chain(self.array_dense,
                         ['min-max', 'pca', 'min-max'],
                         [{'feature_range': (-5, 5)}, {},
                          {'feature_range': (0, 1)}])
        # Expected shape is (3, 3), as pca max n_components is 4-1

    def test_chain_gradient(self):
        """Check gradient of a chain of preprocessors."""
        # Expected shape is (n_feats, ), so (4, )
        self._test_chain_gradient(self.array_dense,
                                  ['min-max', 'mean-std', 'min-max'],
                                  [{'feature_range': (-5, 5)}, {},
                                   {'feature_range': (0, 1)}])


if __name__ == '__main__':
    CNormalizerTestCases.main()
