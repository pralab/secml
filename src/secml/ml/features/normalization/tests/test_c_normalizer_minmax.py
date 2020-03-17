from secml.ml.features.normalization.tests import CNormalizerTestCases

from sklearn.preprocessing import MinMaxScaler

from secml.ml.features.normalization import CNormalizerMinMax


class TestCNormalizerLinear(CNormalizerTestCases):
    """Unittest for CNormalizerLinear."""

    def test_norm_minmax(self):
        """Test for CNormalizerMinMax."""
        self.sklearn_comp(self.array_dense, MinMaxScaler(), CNormalizerMinMax())
        self.sklearn_comp(self.array_sparse, MinMaxScaler(), CNormalizerMinMax())
        self.sklearn_comp(self.row_dense.atleast_2d(), MinMaxScaler(), CNormalizerMinMax())
        self.sklearn_comp(self.row_sparse, MinMaxScaler(), CNormalizerMinMax())
        self.sklearn_comp(self.column_dense, MinMaxScaler(), CNormalizerMinMax())
        self.sklearn_comp(self.column_sparse, MinMaxScaler(), CNormalizerMinMax())
        self.sklearn_comp(self.array_dense*2, MinMaxScaler(), CNormalizerMinMax())
        self.sklearn_comp(self.array_sparse*2, MinMaxScaler(), CNormalizerMinMax())
        self.sklearn_comp(self.row_dense.atleast_2d()*2, MinMaxScaler(), CNormalizerMinMax())
        self.sklearn_comp(self.row_sparse*2, MinMaxScaler(), CNormalizerMinMax())
        self.sklearn_comp(self.column_dense*2, MinMaxScaler(), CNormalizerMinMax())
        self.sklearn_comp(self.column_sparse*2, MinMaxScaler(), CNormalizerMinMax())

    def test_chain(self):
        """Test a chain of preprocessors."""
        feature_range = {'feature_range': (0, 1)}
        self.setup_x_chain('min-max', feature_range)
        # Expected shape is (3, 3), as pca max n_components is 4-1

    def test_chain_gradient(self):
        """Check gradient of a chain of preprocessors."""
        # Expected shape is (n_feats, ), so (4, )
        names = ['min-max', 'mean-std', 'min-max']
        feature_ranges = [{'feature_range': (-5, 5)}, {}, {'feature_range': (0, 1)}]
        self.setup_grad(names, feature_ranges)


if __name__ == '__main__':
    CNormalizerTestCases.main()
