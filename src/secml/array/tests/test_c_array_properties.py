from secml.array.tests import CArrayTestCases

from secml.array import CArray


class TestCArrayProperties(CArrayTestCases):
    """Unit test for CArray PROPERTIES."""

    def test_non_zero_indices(self):
        """Property test non_zero_indices."""
        self.logger.info("Testing non_zero_indices property")

        # FIXME: UPDATE UNITTESTS
        def non_zero_indices(self, structure_name, matrix, row_vector,
                             column_vector):
            self.logger.info("nnz_indices: matrix \n" + str(matrix))
            self.logger.info(
                "Non zero index are: \n" + str(matrix.nnz_indices))
            self.assertEqual(
                matrix.nnz_indices, [[0, 0, 1, 1, 2, 2], [0, 3, 0, 1, 0, 1]])

            self.assertIsInstance(matrix.nnz_indices, list)
            self.assertEqual(2, len(matrix.nnz_indices))
            self.assertTrue(
                all(isinstance(elem, list) for elem in matrix.nnz_indices))

        non_zero_indices(self, "sparse", self.array_sparse, self.row_sparse,
                         self.column_sparse)
        non_zero_indices(self, "dense", self.array_dense, self.row_sparse,
                         self.column_dense)

    def test_nnz(self):
        """Test for CArray.nnz property."""
        self.logger.info("Testing CArray.nnz property")

        def check_nnz(array):
            self.logger.info("array:\n{:}".format(array))
            res = array.nnz
            self.logger.info("nnz: {:}".format(res))
            self.assertIsInstance(res, int)
            self.assertEqual(array.nnz_data.size, res)

        check_nnz(self.array_sparse)
        check_nnz(self.row_sparse)
        check_nnz(self.column_sparse)
        check_nnz(self.array_dense)
        check_nnz(self.row_dense)
        check_nnz(self.column_dense)

    def test_nnz_data(self):
        """Test for CArray.nnz_data property."""

        def check_nnz_data(array, expected_nnz):
            self.logger.info("array:\n{:}".format(array))

            res = array.nnz_data
            self.logger.info("nnz_data:\n{:}".format(array.nnz_data))
            self.assertIsInstance(res, CArray)
            self.assertEqual(1, res.ndim)
            self.assertEqual(array.nnz, res.size)
            self.assertFalse((array.nnz_data != expected_nnz).any())

        self.logger.info("Testing CArray.nnz_data property")
        check_nnz_data(self.array_sparse, CArray([1, 5, 2, 4, 3, 6]))
        check_nnz_data(self.row_sparse, CArray([4, 6]))
        check_nnz_data(self.column_sparse, CArray([4, 6]))
        check_nnz_data(self.array_dense, CArray([1, 5, 2, 4, 3, 6]))
        check_nnz_data(self.row_dense, CArray([4, 6]))
        check_nnz_data(self.column_dense, CArray([4, 6]))
    

if __name__ == '__main__':
    CArrayTestCases.main()
