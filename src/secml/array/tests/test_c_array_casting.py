from secml.array.tests import CArrayTestCases

import numpy as np
import scipy.sparse as scs


class TestCArrayCasting(CArrayTestCases):
    """Unit test for CArray CASTING methods."""

    def test_tondarray(self):
        """Test for CArray.tondarray() method."""
        self.logger.info("Test for CArray.tondarray() method.")

        def _check_tondarray(array):
            self.logger.info("array:\n{:}".format(array))

            ndarray = array.tondarray()
            self.logger.info("array.tondarray():\n{:}".format(ndarray))

            self.assertIsInstance(ndarray, np.ndarray)

            self.assertEqual(array.size, ndarray.size)
            self.assertEqual(array.shape, ndarray.shape)

        # Sparse arrays
        _check_tondarray(self.array_sparse)
        _check_tondarray(self.row_sparse)
        _check_tondarray(self.column_sparse)

        # Dense arrays
        _check_tondarray(self.array_dense)
        _check_tondarray(self.row_flat_dense)
        _check_tondarray(self.row_dense)
        _check_tondarray(self.column_dense)

        # Bool arrays
        _check_tondarray(self.array_dense_bool)
        _check_tondarray(self.array_sparse_bool)

        _check_tondarray(self.single_flat_dense)
        _check_tondarray(self.single_dense)
        _check_tondarray(self.single_sparse)

        _check_tondarray(self.empty_dense)
        _check_tondarray(self.empty_flat_dense)
        _check_tondarray(self.empty_sparse)

    def test_toscs(self):
        """Test for CArray.tocsr(), CArray.tocoo(), CArray.tocsc(),
        CArray.todia(), CArray.todok(), CArray.tolil() methods."""
        # Will test conversion from dense and between each sparse format
        scs_formats = (
            ('csr', scs.csr_matrix),
            ('coo', scs.coo_matrix),
            ('csc', scs.csc_matrix),
            ('dia', scs.dia_matrix),
            ('dok', scs.dok_matrix),
            ('lil', scs.lil_matrix)
        )
        for scs_format, scs_type in scs_formats:
            self.logger.info(
                "Test for CArray.to{:}() method.".format(scs_format))

            def _check_conversion(array):
                self.logger.info("array:\n{:}".format(array))
                if array.issparse:
                    self.logger.info("array sparse format: {:}".format(
                        array._data._data.getformat()))

                res = getattr(array, 'to{:}'.format(scs_format))()
                self.logger.info(
                    "array.to{:}():\n{:}".format(scs_format, res))
                self.logger.info(
                    "result sparse format: {:}".format(res.getformat()))

                self.assertIsInstance(res, scs_type)

                # size returns the nnz for sparse arrays, DO NOT USE IT
                self.assertEqual(array.size, res.shape[0] * res.shape[1])

                if array.isdense:  # flat dense arrays become 2D when sparse
                    self.assertEqual(array.atleast_2d().shape, res.shape)

            # Sparse arrays
            # Checking conversion from default sparse format (csr)
            _check_conversion(self.array_sparse)
            _check_conversion(self.row_sparse)
            _check_conversion(self.column_sparse)
            # Inner loop to check between formats conversion
            for scs_format_start, _ in scs_formats:
                self.array_sparse._data._data = getattr(
                    self.array_sparse, 'to{:}'.format(scs_format_start))()
                _check_conversion(self.array_sparse)
                self.row_sparse._data._data = getattr(
                    self.row_sparse, 'to{:}'.format(scs_format_start))()
                _check_conversion(self.row_sparse)
                self.column_sparse._data._data = getattr(
                    self.column_sparse, 'to{:}'.format(scs_format_start))()
                _check_conversion(self.column_sparse)

            # Dense arrays
            _check_conversion(self.array_dense)
            _check_conversion(self.row_flat_dense)
            _check_conversion(self.row_dense)
            _check_conversion(self.column_dense)

            # Bool arrays
            _check_conversion(self.array_dense_bool)
            _check_conversion(self.array_sparse_bool)

            _check_conversion(self.single_flat_dense)
            _check_conversion(self.single_dense)
            _check_conversion(self.single_sparse)

            _check_conversion(self.empty_dense)
            _check_conversion(self.empty_flat_dense)
            _check_conversion(self.empty_sparse)

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
