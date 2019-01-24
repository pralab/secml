from secml.utils import CUnitTest

from sklearn.preprocessing import StandardScaler

from secml.array import CArray
from secml.ml.features.normalization import CNormalizerMeanSTD


class TestCNormalizerMeanStd(CUnitTest):
    """Unittest for CNormalizerMeanStd"""

    def setUp(self):
        """Code to run before each test."""
        self.array_dense = CArray([[1, 0, 0, 5],
                                   [2, 4, 0, 0],
                                   [3, 6, 0, 0]])
        self.array_sparse = CArray(self.array_dense.deepcopy(), tosparse=True)

        self.row_dense = CArray([4, 0, 6])
        self.column_dense = self.row_dense.deepcopy().T

        self.row_sparse = CArray(self.row_dense.deepcopy(), tosparse=True)
        self.column_sparse = self.row_sparse.deepcopy().T

    def test_zscore(self):
        """Test for CNormalizerMeanStd to obtain zero mean and unit variance"""

        def sklearn_comp(array):

            self.logger.info("Original array is:\n{:}".format(array))

            # Sklearn normalizer
            target = CArray(StandardScaler().fit_transform(
                array.astype(float).tondarray())).round(4)
            # Our normalizer
            n = CNormalizerMeanSTD().fit(array)
            result = n.normalize(array).round(4)

            self.logger.info("Correct result is:\n{:}".format(target))
            self.logger.info("Our result is:\n{:}".format(result))

            self.assertFalse((target != result).any(),
                             "\n{:}\nis different from target\n"
                             "{:}".format(result, target))

            self.logger.info("Testing without std")
            # Sklearn normalizer
            target = CArray(StandardScaler(with_std=False).fit_transform(
                array.astype(float).tondarray())).round(4)
            # Our normalizer
            n = CNormalizerMeanSTD(with_std=False).fit(array)
            result = n.normalize(array).round(4)

            self.logger.info("Correct result is:\n{:}".format(target))
            self.logger.info("Our result is:\n{:}".format(result))

            self.assertFalse((target != result).any(),
                             "\n{:}\nis different from target\n"
                             "{:}".format(result, target))

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

                n = CNormalizerMeanSTD(mean=mean, std=std).fit(array)
                out = n.normalize(array)

                self.logger.info("Result is:\n{:}".format(out))

                out_mean = out.mean(axis=0, keepdims=False)
                out_std = out.std(axis=0, keepdims=False)

                self.logger.info("Result mean is:\n{:}".format(out_mean))
                self.logger.info("Result std is:\n{:}".format(out_std))

                rev = n.revert(out).round(4)
                self.assertFalse((array != rev).any(),
                                 "Reverted array not equal to original")


if __name__ == '__main__':
    CUnitTest.main()
