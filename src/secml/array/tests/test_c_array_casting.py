from secml.array.tests import CArrayTestCases

import numpy as np
import scipy.sparse as scs


class TestCArrayCasting(CArrayTestCases):
    """Unit test for CArray CASTING methods."""

    def test_tondarray_tocsr(self):
        """Test for CArray.tondarray(), CArray.tocsr() method."""
        self.logger.info("Test for CArray.tondarray(), CArray.tocsr() method.")

        def _check_tondarray_tocsr(array):
            self.logger.info("array:\n{:}".format(array))

            ndarray = array.tondarray()
            self.logger.info("array.tondarray():\n{:}".format(ndarray))
            csr = array.tocsr()
            self.logger.info("array.tocsr():\n{:}".format(csr))

            self.assertIsInstance(ndarray, np.ndarray)
            self.assertIsInstance(csr, scs.csr_matrix)

            self.assertEqual(array.size, ndarray.size)
            self.assertEqual(array.size, csr.shape[0] * csr.shape[1])

            self.assertEqual(array.shape, ndarray.shape)
            if array.isdense:  # flat dense arrays become 2D when sparse
                self.assertEqual(array.atleast_2d().shape, csr.shape)

        # Sparse arrays
        _check_tondarray_tocsr(self.array_sparse)
        _check_tondarray_tocsr(self.row_sparse)
        _check_tondarray_tocsr(self.column_sparse)

        # Dense arrays
        _check_tondarray_tocsr(self.array_dense)
        _check_tondarray_tocsr(self.row_flat_dense)
        _check_tondarray_tocsr(self.row_dense)
        _check_tondarray_tocsr(self.column_dense)

        # Bool arrays
        _check_tondarray_tocsr(self.array_dense_bool)
        _check_tondarray_tocsr(self.array_sparse_bool)

        _check_tondarray_tocsr(self.single_flat_dense)
        _check_tondarray_tocsr(self.single_dense)
        _check_tondarray_tocsr(self.single_sparse)

        _check_tondarray_tocsr(self.empty_dense)
        _check_tondarray_tocsr(self.empty_flat_dense)
        _check_tondarray_tocsr(self.empty_sparse)

    def test_tolist(self):
        """Test for CArray.tolist() method."""
        self.logger.info("Test for CArray.tolist() method.")

        def _check_tolist(array):
            self.logger.info("array:\n{:}".format(array))

            array_list = array.tolist()
            self.logger.info("array.tolist():\n{:}".format(array_list))

            self.assertIsInstance(array_list, list)

            self.assertEqual(len(array_list), array.shape[0])
            if array.ndim > 1:
                for elem in array_list:
                    self.assertEqual(len(elem), array.shape[1])

        # Sparse arrays
        _check_tolist(self.array_sparse)
        _check_tolist(self.row_sparse)
        _check_tolist(self.column_sparse)

        # Dense arrays
        _check_tolist(self.array_dense)
        _check_tolist(self.row_flat_dense)
        _check_tolist(self.row_dense)
        _check_tolist(self.column_dense)

        # Bool arrays
        _check_tolist(self.array_dense_bool)
        _check_tolist(self.array_sparse_bool)

        _check_tolist(self.single_flat_dense)
        _check_tolist(self.single_dense)
        _check_tolist(self.single_sparse)

        _check_tolist(self.empty_dense)
        _check_tolist(self.empty_flat_dense)
        _check_tolist(self.empty_sparse)

    def test_todense(self):
        """Test for CArray.todense() method."""
        self.logger.info("Test for CArray.todense() method")

        def _check_todense(array):
            self.logger.info("array:\n{:}".format(array))

            array_dense = array.todense()
            self.logger.info("array.todense():\n{:}".format(array_dense))

            self.assertTrue(array_dense.isdense)
            self.assertEqual(array_dense.size, array.size)
            self.assertEqual(array_dense.shape, array.shape)

        # Sparse arrays
        _check_todense(self.array_sparse)
        _check_todense(self.row_sparse)
        _check_todense(self.column_sparse)

        # Dense arrays
        _check_todense(self.array_dense)
        _check_todense(self.row_flat_dense)
        _check_todense(self.row_dense)
        _check_todense(self.column_dense)

        # Bool arrays
        _check_todense(self.array_dense_bool)
        _check_todense(self.array_sparse_bool)

        _check_todense(self.single_flat_dense)
        _check_todense(self.single_dense)
        _check_todense(self.single_sparse)

        _check_todense(self.empty_dense)
        _check_todense(self.empty_flat_dense)
        _check_todense(self.empty_sparse)

        with self.assertRaises(ValueError):
            self.array_dense.todense(dtype=int)
        with self.assertRaises(ValueError):
            self.array_dense.todense(shape=())
        with self.assertRaises(ValueError):
            self.array_dense.todense(dtype=int, shape=())

    def test_tosparse(self):
        """Test for CArray.tosparse() method."""
        self.logger.info("Test for CArray.tosparse() method")

        def _check_tosparse(array):
            self.logger.info("array:\n{:}".format(array))

            array_sparse = array.tosparse()
            self.logger.info("array.tosparse():\n{:}".format(array_sparse))

            self.assertTrue(array_sparse.issparse)
            self.assertEqual(array_sparse.size, array.size)
            self.assertEqual(array_sparse.is_vector_like, array.is_vector_like)

            if array.ndim > 1:  # original array is 2D, shape must not change
                self.assertEqual(array_sparse.shape, array.shape)

        # Sparse arrays
        _check_tosparse(self.array_sparse)
        _check_tosparse(self.row_sparse)
        _check_tosparse(self.column_sparse)

        # Dense arrays
        _check_tosparse(self.array_dense)
        _check_tosparse(self.row_flat_dense)
        _check_tosparse(self.row_dense)
        _check_tosparse(self.column_dense)

        # Bool arrays
        _check_tosparse(self.array_dense_bool)
        _check_tosparse(self.array_sparse_bool)

        _check_tosparse(self.single_flat_dense)
        _check_tosparse(self.single_dense)
        _check_tosparse(self.single_sparse)

        _check_tosparse(self.empty_dense)
        _check_tosparse(self.empty_flat_dense)
        _check_tosparse(self.empty_sparse)

        with self.assertRaises(ValueError):
            self.array_sparse.tosparse(dtype=int)
        with self.assertRaises(ValueError):
            self.array_sparse.tosparse(shape=())
        with self.assertRaises(ValueError):
            self.array_sparse.tosparse(dtype=int, shape=())
    

if __name__ == '__main__':
    CArrayTestCases.main()
