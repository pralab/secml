import numpy as np

from secml.testing import CUnitTest
from secml.utils import fm
from secml.array.c_dense import CDense
from secml.array.c_sparse import CSparse


class TestCSparse(CUnitTest):
    """Unit test for CSparse."""

    def setUp(self):
        """Basic set up."""
        self.dense = CDense([[1, 0, 0, 0, 5],
                             [2, 4, 0, 0, 0],
                             [3, 4, 5, 0, 0]])
        self.dense_vector = CDense([1, 0, 0, 0, 3])
        self.sparse_matrix = CSparse(self.dense)
        self.sparse_vector = CSparse(self.dense_vector)

    def test_save_load(self):
        """Test save/load of sparse arrays"""
        self.logger.info("UNITTEST - CSparse - save/load")

        test_file = fm.join(fm.abspath(__file__), 'test.txt')

        # Cleaning test file
        try:
            fm.remove_file(test_file)
        except (OSError, IOError) as e:
            if e.errno != 2:
                raise e

        self.logger.info(
            "UNITTEST - CSparse - Testing save/load for sparse matrix")

        self.sparse_matrix.save(test_file)

        self.logger.info(
            "Saving again with overwrite=False... IOError should be raised.")
        with self.assertRaises(IOError) as e:
            self.sparse_matrix.save(test_file)
        self.logger.info(e.exception)

        loaded_sparse_matrix = CSparse.load(test_file, dtype=int)

        self.assertFalse((loaded_sparse_matrix != self.sparse_matrix).any(),
                         "Saved and loaded arrays (matrices) are not equal!")

        self.logger.info(
            "UNITTEST - CSparse - Testing save/load for sparse vector")

        self.sparse_vector.save(test_file, overwrite=True)
        loaded_sparse_vector = CSparse.load(test_file, dtype=int)

        self.assertFalse((loaded_sparse_vector != self.sparse_vector).any(),
                         "Saved and loaded arrays (vectors) are not equal!")

        # Cleaning test file
        try:
            fm.remove_file(test_file)
        except (OSError, IOError) as e:
            if e.errno != 2:
                raise e

    def test_mixed(self):

        print(self.sparse_matrix[np.ravel(0)[0], np.ravel(0)[0]])
        print(type(self.sparse_matrix[np.ravel(0)[0], np.ravel(0)[0]]))


if __name__ == '__main__':
    CUnitTest.main()
