from secml.ml.features.normalization.tests import CNormalizerTestCases

from sklearn.preprocessing import StandardScaler

from secml.ml.features.normalization import CNormalizerMeanStd


class TestCNormalizerMeanStd(CNormalizerTestCases):
    """Unittests for CNormalizerMeanStd."""

    def test_transform(self):
        """Test for `.transform()` method."""
        for with_std in (True, False):

            self.logger.info("Testing using std? {:}".format(with_std))

            self._sklearn_comp(self.array_dense,
                               StandardScaler(with_std=with_std),
                               CNormalizerMeanStd(with_std=with_std))
            self._sklearn_comp(self.array_sparse,
                               StandardScaler(with_std=with_std),
                               CNormalizerMeanStd(with_std=with_std))
            self._sklearn_comp(self.row_dense.atleast_2d(),
                               StandardScaler(with_std=with_std),
                               CNormalizerMeanStd(with_std=with_std))
            self._sklearn_comp(self.row_sparse,
                               StandardScaler(with_std=with_std),
                               CNormalizerMeanStd(with_std=with_std))
            self._sklearn_comp(self.column_dense,
                               StandardScaler(with_std=with_std),
                               CNormalizerMeanStd(with_std=with_std))
            self._sklearn_comp(self.column_sparse,
                               StandardScaler(with_std=with_std),
                               CNormalizerMeanStd(with_std=with_std))

    def test_mean_std(self):
        """Test using specific mean/std."""
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
        """Test a chain of preprocessors."""
        self._test_chain(self.array_dense,
                         ['min-max', 'pca', 'mean-std'],
                         [{'feature_range': (-5, 5)}, {}, {}])

    def test_chain_gradient(self):
        """Check gradient of a chain of preprocessors."""
        self._test_chain_gradient(self.array_dense,
                                  ['min-max', 'mean-std'],
                                  [{'feature_range': (-5, 5)}, {}])


if __name__ == '__main__':
    CNormalizerTestCases.main()
