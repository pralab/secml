"""
This is the class for testing CCOlumnsNormalizer class

@author: Ambra Demontis, Marco Melis

When adding a test method please use method to test name plus 'test_' as suffix.
As first test method line use self.logger.info("UNITTEST - CLASSNAME - METHODNAME")

"""
import unittest

from secml.utils import CUnitTest
from secml.array import CArray
from secml.features.normalization import CNormalizerMinMax, CNormalizerZScore, CNormalizerRow
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer


class TestArrayNormalizers(CUnitTest):
    """Unittest container"""

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
        """Test for ZScoreScaler. This compares sklearn equivalent to our normalizer."""

        def sklearn_comp(array):

            self.logger.info("Original array is:\n{:}".format(array))

            # Sklearn normalizer
            target = CArray(StandardScaler().fit_transform(array.tondarray())).round(4)
            # Our normalizer
            result = CNormalizerZScore().train_normalize(array).round(4)

            self.assertFalse((target != result).any(), "\n{:}\nis different from target\n{:}".format(result, target))

            self.logger.info("Correct result is:\n{:}".format(result))

        sklearn_comp(self.array_dense)
        sklearn_comp(self.array_sparse)
        sklearn_comp(self.row_dense.atleast_2d())  # We manage flat vectors differently from numpy/sklearn
        sklearn_comp(self.row_sparse)
        sklearn_comp(self.column_dense)
        sklearn_comp(self.column_sparse)

    def test_rownormalizer(self):
        """Test for RowNormalizer. This compares sklearn equivalent to our normalizer."""

        def sklearn_comp(array):

            self.logger.info("Original array is:\n{:}".format(array))

            # Sklearn normalizer (requires float dtype input)
            target = CArray(Normalizer().fit_transform(array.astype(float).get_data())).round(4)
            # Our normalizer
            result = CNormalizerRow().train_normalize(array).round(4)

            self.assertFalse((target != result).any(), "\n{:}\nis different from target\n{:}".format(result, target))

            self.logger.info("Correct result is:\n{:}".format(result))

        sklearn_comp(self.array_dense)
        sklearn_comp(self.array_sparse)
        sklearn_comp(self.row_dense.atleast_2d())  # We manage flat vectors differently from numpy/sklearn
        sklearn_comp(self.row_sparse)
        sklearn_comp(self.column_dense)
        sklearn_comp(self.column_sparse)

    def test_minmaxscaler(self):
        """Test for MinMaxScaler. This compares sklearn equivalent to our normalizer."""

        def sklearn_comp(array):

            self.logger.info("Original array is:\n{:}".format(array))

            # Sklearn normalizer (requires float dtype input)
            target = CArray(MinMaxScaler().fit_transform(array.astype(float).tondarray())).round(4)
            # Our normalizer
            result = CNormalizerMinMax().train_normalize(array).round(4)

            self.assertFalse((target != result).any(), "\n{:}\nis different from target\n{:}".format(result, target))

            self.logger.info("Correct result is:\n{:}".format(result))

        sklearn_comp(self.array_dense)
        sklearn_comp(self.row_dense.atleast_2d())  # We manage flat vectors differently from numpy/sklearn
        sklearn_comp(self.column_dense)


if __name__ == '__main__':
    unittest.main()
