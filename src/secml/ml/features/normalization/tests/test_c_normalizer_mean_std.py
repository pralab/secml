from secml.ml.features.normalization.tests import CNormalizerTestCases
from sklearn.preprocessing import StandardScaler
from secml.ml.features.normalization import CNormalizerMeanStd


class TestCNormalizerMeanStd(CNormalizerTestCases):
    """Unittest for CNormalizerMeanStd."""

    def test_zscore(self):
        """Test for CNormalizerMeanStd to obtain zero mean and unit variance."""
        self._sklearn_comp(self.array_dense, StandardScaler(),
                           CNormalizerMeanStd())
        self._sklearn_comp(self.array_sparse, StandardScaler(),
                           CNormalizerMeanStd())
        self._sklearn_comp(self.row_dense.atleast_2d(), StandardScaler(),
                           CNormalizerMeanStd())
        self._sklearn_comp(self.row_sparse, StandardScaler(),
                           CNormalizerMeanStd())
        self._sklearn_comp(self.column_dense, StandardScaler(),
                           CNormalizerMeanStd())
        self._sklearn_comp(self.column_sparse, StandardScaler(),
                           CNormalizerMeanStd())

        self._sklearn_comp(self.array_dense, StandardScaler(with_std=False),
                           CNormalizerMeanStd(with_std=False))
        self._sklearn_comp(self.array_sparse, StandardScaler(with_std=False),
                           CNormalizerMeanStd(with_std=False))
        self._sklearn_comp(self.row_dense.atleast_2d(),
                           StandardScaler(with_std=False),
                           CNormalizerMeanStd(with_std=False))
        self._sklearn_comp(self.row_sparse, StandardScaler(with_std=False),
                           CNormalizerMeanStd(with_std=False))
        self._sklearn_comp(self.column_dense, StandardScaler(with_std=False),
                           CNormalizerMeanStd(with_std=False))
        self._sklearn_comp(self.column_sparse, StandardScaler(with_std=False),
                           CNormalizerMeanStd(with_std=False))

    def test_normalizer_mean_std(self):
        """Test for CNormalizerMeanStd."""
        for (mean, std) in [(1.5, 0.1),
                            ((1.0, 1.1, 1.2, 1.3), (0.0, 0.1, 0.2, 0.3))]:
            for array in [self.array_dense, self.array_sparse]:
                self.logger.info("Original array is:\n{:}".format(array))
                self.logger.info(
                    "Normalizing using mean: {:} std: {:}".format(mean, std))

                n = CNormalizerMeanStd(mean=mean, std=std).fit(array)
                out = n.transform(array)

                self.logger.info("Result is:\n{:}".format(out))

                out_mean = out.mean(axis=0, keepdims=False)
                out_std = out.std(axis=0, keepdims=False)

                self.logger.info("Result mean is:\n{:}".format(out_mean))
                self.logger.info("Result std is:\n{:}".format(out_std))

                rev = n.inverse_transform(out)

                self.assert_array_almost_equal(array, rev)

    def test_chain(self):
        """Tests a chain of preprocessors related to CNormalizerMeanStd."""
        self._test_chain(self.array_dense,
                         ['min-max', 'pca', 'mean-std'],
                         [{'feature_range': (-5, 5)}, {}, {}])
        # Expected shape is (3, 3), as pca max n_components is 4-1

    def test_chain_gradient(self):
        """Tests the gradient of a chain of preprocessors
        related to CNormalizerMeanStd.
        """
        # Expected shape is (n_feats, ), so (4, )
        self._test_chain_gradient(self.array_dense,
                                  ['min-max', 'mean-std'],
                                  [{'feature_range': (-5, 5)}, {}])


if __name__ == '__main__':
    CNormalizerTestCases.main()
