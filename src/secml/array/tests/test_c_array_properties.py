from secml.utils import CUnitTest
from c_array_testcases import CArrayTestCases

from secml.array import CArray


class TestCArrayProperties(CArrayTestCases.TestCArray):
    """Unit test for CArray PROPERTIES."""

    def test_non_zero_indices(self):
        """Property test non_zero_indices."""
        self.logger.info("Testing non_zero_indices property")

        def non_zero_indices(self, structure_name, matrix, row_vector,
                             column_vector):
            self.logger.info("nnz_indices: matrix \n" + str(matrix))
            self.logger.info(
                "Non zero index are: \n" + str(matrix.nnz_indices))
            self.assertEquals(
                matrix.nnz_indices == [[0, 0, 1, 1, 2, 2], [0, 3, 0, 1, 0, 1]],
                True, "nnz_indices returned the wrong indices indices")

            self.assertEquals(isinstance(matrix.nnz_indices, list), True,
                              "nnz_indices not returned a list")
            self.assertEquals(len(matrix.nnz_indices), 2,
                              "nnz_indices not returned a list of 2 element")
            self.assertEquals(
                all(isinstance(elem, list) for elem in matrix.nnz_indices),
                True,
                "nnz_indices not returned a list of 2 lists")

        non_zero_indices(self, "sparse", self.array_sparse, self.row_sparse,
                         self.column_sparse)
        non_zero_indices(self, "dense", self.array_dense, self.row_sparse,
                         self.column_dense)

    def test_non_zero(self):
        """Test for CArray.nnz, CArray.nnz_data properties."""
        self.logger.info("Testing CArray.nnz property")

        def check_nnz(array):
            self.logger.info("array:\n{:}".format(array))
            self.logger.info("nnz: {:}".format(array.nnz))
            self.assertEquals(array.nnz, array.nnz_data.size)

        check_nnz(self.array_sparse)
        check_nnz(self.row_sparse)
        check_nnz(self.column_sparse)
        check_nnz(self.array_dense)
        check_nnz(self.row_dense)
        check_nnz(self.column_dense)

        def check_nnz_data(array, expected_nnz):
            self.logger.info("array:\n{:}".format(array))
            self.logger.info("nnz: {:}".format(array.nnz))
            self.assertFalse((array.nnz_data != expected_nnz).any())

        self.logger.info("Testing CArray.nnz_data property")
        check_nnz_data(self.array_sparse, CArray([1, 5, 2, 4, 3, 6]))
        check_nnz_data(self.row_sparse, CArray([4, 6]))
        check_nnz_data(self.column_sparse, CArray([4, 6]))
        check_nnz_data(self.array_dense, CArray([1, 5, 2, 4, 3, 6]))
        check_nnz_data(self.row_dense, CArray([4, 6]))
        check_nnz_data(self.column_dense, CArray([4, 6]))
    

if __name__ == '__main__':
    CUnitTest.main()
