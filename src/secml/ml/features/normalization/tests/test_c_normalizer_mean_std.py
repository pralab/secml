from secml.ml.features.tests import CPreProcessTestCases

from sklearn.preprocessing import StandardScaler

from secml.array import CArray
from secml.ml.features.normalization import CNormalizerMeanStd


class TestCNormalizerMeanStd(CPreProcessTestCases):
    """Unittest for CNormalizerMeanStd"""

    def test_zscore(self):
        """Test for CNormalizerMeanStd to obtain zero mean and unit variance"""

        def sklearn_comp(array):

            self.logger.info("Original array is:\n{:}".format(array))

            # Sklearn normalizer
            target = CArray(StandardScaler().fit_transform(
                array.astype(float).tondarray()))
            # Our normalizer
            n = CNormalizerMeanStd().fit(array)
            result = n.transform(array)

            self.logger.info("Correct result is:\n{:}".format(target))
            self.logger.info("Our result is:\n{:}".format(result))

            self.assert_array_almost_equal(target, result)

            self.logger.info("Testing without std")
            # Sklearn normalizer
            target = CArray(StandardScaler(with_std=False).fit_transform(
                array.astype(float).tondarray()))
            # Our normalizer
            n = CNormalizerMeanStd(with_std=False).fit(array)
            result = n.transform(array)

            self.logger.info("Correct result is:\n{:}".format(target))
            self.logger.info("Our result is:\n{:}".format(result))

            self.assert_array_almost_equal(target, result)

        sklearn_comp(self.array_dense)
        sklearn_comp(self.array_sparse)
        sklearn_comp(self.row_dense.atleast_2d())
        sklearn_comp(self.row_sparse)
        sklearn_comp(self.column_dense)
        sklearn_comp(self.column_sparse)

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
        """Test a chain of preprocessors."""
        x_chain = self._test_chain(
            self.array_dense,
            ['min-max', 'pca', 'mean-std'],
            [{'feature_range': (-5, 5)}, {}, {}]
        )

        # Expected shape is (3, 3), as pca max n_components is 4-1
        self.assertEqual((self.array_dense.shape[0],
                          self.array_dense.shape[1]-1), x_chain.shape)

    def test_chain_gradient(self):
        """Check gradient of a chain of preprocessors."""
        grad = self._test_chain_gradient(
            self.array_dense,
            ['min-max', 'mean-std'],
            [{'feature_range': (-5, 5)}, {}]
        )

        # Expected shape is (n_feats, ), so (4, )
        self.assertEqual((self.array_dense.shape[1], ), grad.shape)


if __name__ == '__main__':
    CPreProcessTestCases.main()
