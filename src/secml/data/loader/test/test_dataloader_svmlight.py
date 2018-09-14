"""
Created on 04/mag/2015

@author: Davide Maiorca, Ambra Demontis
"""
import unittest
from secml.utils import CUnitTest

from secml.data import CDataset
from secml.data.loader import CDataLoaderSvmLight
from secml.array import CArray, Csparse, Cdense


class TestCDataLoaderSvmLight(CUnitTest):
    """Unit test for CEvasion."""

    def setUp(self):

        self.dataloader = CDataLoaderSvmLight()
        self.patterns = CArray([[1, 0, 2], [4, 0, 5]], tosparse=True)
        self.labels = CArray([0, 1])

    def _non_zero_columns_search(array):
        """
        Given an array return a CArray with non zero column index
        """
        col_num = array.shape[1]
        non_zero_col = CArray([], dtype=int)
        for c in xrange(col_num):
            col = array[:, c]
            if col.any() == True:
                non_zero_col = non_zero_col.append(c)

        return non_zero_col

    def test_save_and_load_svmlight_file(self):
        """Testing libsvm dataset loading and saving."""
        self.logger.info("Testing libsvm dataset loading and saving...")

        self.logger.info("Patterns saved:\n{:}".format(self.patterns))
        self.logger.info("Labels saved:\n{:}".format(self.labels))

        CDataLoaderSvmLight.dump(
            CDataset(self.patterns, self.labels), "myfile.libsvm")

        new_dataset = CDataLoaderSvmLight().load("myfile.libsvm")

        self.assertFalse((new_dataset.X != self.patterns).any())
        self.assertFalse((new_dataset.Y != self.labels).any())

        # load data but now remove all zero features (colums)
        new_dataset = CDataLoaderSvmLight().load(
            "myfile.libsvm", remove_all_zero=True)

        self.logger.info("Patterns loaded:\n{:}".format(new_dataset.X))
        self.logger.info("Labels loaded:\n{:}".format(new_dataset.Y))
        self.logger.info(
            "Mapping back:\n{:}".format(new_dataset.idx_mapping))

        self.assertTrue(new_dataset.X.issparse)
        self.assertTrue(new_dataset.Y.isdense)
        self.assertTrue(new_dataset.idx_mapping.isdense)

        # non-zero elements should be unchanged
        self.assertEquals(self.patterns.nnz, new_dataset.X.nnz)
        new_nnz_data = new_dataset.X.nnz_data
        new_nnz_data.sort()
        self.assertFalse((self.patterns.nnz_data != new_nnz_data).any())

        # With idx_mapping we should be able to reconstruct original data
        original = CArray.zeros(self.patterns.shape, sparse=True)
        original[:, new_dataset.idx_mapping] = new_dataset.X
        self.assertFalse((self.patterns != original).any())


if __name__ == '__main__':
    unittest.main()
