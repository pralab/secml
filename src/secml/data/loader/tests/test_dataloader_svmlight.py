from secml.testing import CUnitTest

from secml.data import CDataset
from secml.data.loader import CDataLoaderSvmLight
from secml.array import CArray
from secml.utils import fm


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
        for c in range(col_num):
            col = array[:, c]
            if col.any() == True:
                non_zero_col = non_zero_col.append(c)

        return non_zero_col

    def test_save_and_load_svmlight_file(self):
        """Testing libsvm dataset loading and saving."""
        self.logger.info("Testing libsvm dataset loading and saving...")

        test_file = fm.join(fm.abspath(__file__), "myfile.libsvm")

        # Cleaning test file
        try:
            fm.remove_file(test_file)
        except (OSError, IOError) as e:
            if e.errno != 2:
                raise e

        self.logger.info("Patterns saved:\n{:}".format(self.patterns))
        self.logger.info("Labels saved:\n{:}".format(self.labels))

        CDataLoaderSvmLight.dump(
            CDataset(self.patterns, self.labels), test_file)

        new_dataset = CDataLoaderSvmLight().load(test_file)

        self.assertFalse((new_dataset.X != self.patterns).any())
        self.assertFalse((new_dataset.Y != self.labels).any())

        # load data but now remove all zero features (colums)
        new_dataset = CDataLoaderSvmLight().load(
            test_file, remove_all_zero=True)

        self.logger.info("Patterns loaded:\n{:}".format(new_dataset.X))
        self.logger.info("Labels loaded:\n{:}".format(new_dataset.Y))
        self.logger.info(
            "Mapping back:\n{:}".format(new_dataset.header.idx_mapping))

        self.assertTrue(new_dataset.X.issparse)
        self.assertTrue(new_dataset.Y.isdense)
        self.assertTrue(new_dataset.header.idx_mapping.isdense)

        # non-zero elements should be unchanged
        self.assertEqual(self.patterns.nnz, new_dataset.X.nnz)
        new_nnz_data = new_dataset.X.nnz_data
        self.assertFalse((self.patterns.nnz_data != new_nnz_data.sort()).any())

        # With idx_mapping we should be able to reconstruct original data
        original = CArray.zeros(self.patterns.shape, sparse=True)
        original[:, new_dataset.header.idx_mapping] = new_dataset.X
        self.assertFalse((self.patterns != original).any())

        # Cleaning test file
        try:
            fm.remove_file(test_file)
        except (OSError, IOError) as e:
            if e.errno != 2:
                raise e


if __name__ == '__main__':
    CUnitTest.main()
