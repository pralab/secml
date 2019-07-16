from secml.array.tests import CArrayTestCases

from secml.array import CArray


class TestCArrayUtilsComparison(CArrayTestCases):
    """Unit test for CArray UTILS - COMPARISON methods."""

    def test_logical_and(self):
        """Test for CArray.logical_and() method."""
        self.logger.info("Test for CArray.logical_and() method.")

        def _logical_and(array1, array2, expected):
            self.logger.info("a1: \n{:}".format(array1))
            self.logger.info("a2: \n{:}".format(array2))

            logical_and_res = array1.logical_and(array2)
            self.logger.info(
                "a1.logical_and(a2): \n{:}".format(logical_and_res))

            self.assert_array_equal(logical_and_res, expected)

            if array1.issparse or array2.issparse:
                # If a sparse array is involved, result must be sparse
                self.assertTrue(logical_and_res.issparse)

        _logical_and(self.array_sparse, self.array_dense,
                     self.array_sparse.astype(bool))
        _logical_and(self.row_sparse, self.row_dense,
                     self.row_sparse.astype(bool))
        _logical_and(self.column_sparse, self.column_dense,
                     self.column_sparse.astype(bool))
        _logical_and(self.array_dense, self.array_sparse,
                     self.array_dense.astype(bool))
        _logical_and(self.row_dense, self.row_sparse,
                     self.row_dense.astype(bool))
        _logical_and(self.column_dense, self.column_sparse,
                     self.column_dense.astype(bool))

        _logical_and(self.array_sparse, self.array_sparse_nozero,
                     self.array_sparse.astype(bool))
        _logical_and(self.array_dense, self.array_dense_nozero,
                     self.array_dense.astype(bool))
        _logical_and(self.array_sparse, self.array_sparse_allzero,
                     self.array_sparse_allzero.astype(bool))
        _logical_and(self.array_dense, self.array_dense_allzero,
                     self.array_dense_allzero.astype(bool))
        _logical_and(self.array_sparse_allzero, self.array_sparse_allzero,
                     self.array_sparse_allzero.astype(bool))
        _logical_and(self.array_dense_allzero, self.array_dense_allzero,
                     self.array_dense_allzero.astype(bool))

        _logical_and(self.array_sparse_bool, self.array_sparse_bool_true,
                     self.array_sparse_bool.astype(bool))
        _logical_and(self.array_dense_bool, self.array_dense_bool_true,
                     self.array_dense_bool.astype(bool))
        _logical_and(self.array_sparse_bool_false,
                     self.array_sparse_bool_false,
                     self.array_sparse_bool_false.astype(bool))
        _logical_and(self.array_dense_bool_false, self.array_dense_bool_false,
                     self.array_dense_bool_false.astype(bool))

        _logical_and(self.empty_sparse, self.empty_sparse, self.empty_sparse)
        _logical_and(self.empty_flat_dense, self.empty_flat_dense,
                     self.empty_flat_dense)

    def test_logical_or(self):
        """Test for CArray.logical_or() method."""
        self.logger.info("Test for CArray.logical_or() method.")

        def _logical_or(array1, array2, expected):
            self.logger.info("a1: \n{:}".format(array1))
            self.logger.info("a2: \n{:}".format(array2))

            logical_or_res = array1.logical_or(array2)
            self.logger.info("a1.logical_or(a2): \n{:}".format(logical_or_res))

            self.assert_array_equal(logical_or_res, expected)

        _logical_or(self.array_sparse, self.array_dense,
                    self.array_sparse.astype(bool))
        _logical_or(self.row_sparse, self.row_dense,
                    self.row_sparse.astype(bool))
        _logical_or(self.column_sparse, self.column_dense,
                    self.column_sparse.astype(bool))
        _logical_or(self.array_dense, self.array_sparse,
                    self.array_dense.astype(bool))
        _logical_or(self.row_dense, self.row_sparse,
                    self.row_dense.astype(bool))
        _logical_or(self.column_dense, self.column_sparse,
                    self.column_dense.astype(bool))

        _logical_or(self.array_sparse, self.array_sparse_nozero,
                    self.array_sparse_nozero.astype(bool))
        _logical_or(self.array_dense, self.array_dense_nozero,
                    self.array_dense_nozero.astype(bool))
        _logical_or(self.array_sparse, self.array_sparse_allzero,
                    self.array_sparse.astype(bool))
        _logical_or(self.array_dense, self.array_dense_allzero,
                    self.array_sparse.astype(bool))
        _logical_or(self.array_sparse_allzero, self.array_sparse_allzero,
                    self.array_sparse_allzero.astype(bool))
        _logical_or(self.array_dense_allzero, self.array_dense_allzero,
                    self.array_dense_allzero.astype(bool))

        _logical_or(self.array_sparse_bool, self.array_sparse_bool_true,
                    self.array_sparse_bool_true.astype(bool))
        _logical_or(self.array_dense_bool, self.array_dense_bool_true,
                    self.array_dense_bool_true.astype(bool))
        _logical_or(self.array_sparse_bool_false, self.array_sparse_bool_false,
                    self.array_sparse_bool_false.astype(bool))
        _logical_or(self.array_dense_bool_false, self.array_dense_bool_false,
                    self.array_dense_bool_false.astype(bool))

        _logical_or(self.empty_sparse, self.empty_sparse, self.empty_sparse)
        _logical_or(self.empty_flat_dense, self.empty_flat_dense,
                    self.empty_flat_dense)

    def test_logical_not(self):
        """Test for CArray.logical_not() method."""
        self.logger.info("Test for CArray.logical_not() method.")

        def _logical_not(array, expected):
            self.logger.info("a: \n{:}".format(array))

            logical_not_res = array.logical_not()
            self.logger.info("a.logical_not(): \n{:}".format(logical_not_res))

            self.assert_array_equal(logical_not_res, expected)

        _logical_not(self.array_sparse_nozero,
                     self.array_sparse_allzero.astype(bool))
        _logical_not(self.array_dense_nozero,
                     self.array_dense_allzero.astype(bool))
        _logical_not(self.array_sparse_allzero,
                     self.array_sparse_nozero.astype(bool))
        _logical_not(self.array_dense_allzero,
                     self.array_dense_nozero.astype(bool))

        _logical_not(self.array_sparse_bool_false, self.array_sparse_bool_true)
        _logical_not(self.array_dense_bool_false, self.array_sparse_bool_true)
        _logical_not(self.array_sparse_bool_true, self.array_sparse_bool_false)
        _logical_not(self.array_dense_bool_true, self.array_dense_bool_false)

        _logical_not(self.empty_sparse, self.empty_sparse)
        _logical_not(self.empty_flat_dense, self.empty_flat_dense)

    def test_maximum(self):
        """Test for CArray.maximum() method."""
        self.logger.info("Test for CArray.maximum() method.")

        def _maximum(array1, array2):
            array2 = array2.deepcopy()
            # Multiply each nonzero element of 2nd array to 2
            array2[array2 != 0] *= 2

            self.logger.info("a1: \n{:}".format(array1))
            self.logger.info("a2: \n{:}".format(array2))

            maximum_res = array1.maximum(array2)
            self.logger.info("a1.maximum(a2): \n{:}".format(maximum_res))

            if array1.ndim == 2 or array2.ndim == 2:
                array2 = array2.atleast_2d()
            self.assert_array_equal(maximum_res, array2)

        _maximum(self.array_sparse, self.array_sparse)
        _maximum(self.array_dense, self.array_dense)
        _maximum(self.array_sparse, self.array_dense)
        _maximum(self.array_dense, self.array_sparse)
        with self.assertRaises(ValueError):
            _maximum(self.array_sparse, self.array_sparse_sym)
        with self.assertRaises(ValueError):
            _maximum(self.array_dense, self.array_dense_sym)

        _maximum(self.row_flat_dense, self.row_flat_dense)
        _maximum(self.row_flat_dense, self.row_sparse)
        _maximum(self.row_flat_dense, self.row_dense)
        with self.assertRaises(ValueError):
            _maximum(self.row_flat_dense, self.column_dense)

        _maximum(self.row_sparse, self.row_sparse)
        _maximum(self.row_sparse, self.row_flat_dense)
        _maximum(self.row_sparse, self.row_dense)
        with self.assertRaises(ValueError):
            _maximum(self.row_sparse, self.column_sparse)

        _maximum(self.row_dense, self.row_dense)
        _maximum(self.row_dense, self.row_flat_dense)
        _maximum(self.row_dense, self.row_sparse)
        with self.assertRaises(ValueError):
            _maximum(self.row_dense, self.column_dense)

        _maximum(self.column_dense, self.column_dense)
        _maximum(self.column_dense, self.column_sparse)
        _maximum(self.column_sparse, self.column_sparse)
        _maximum(self.column_sparse, self.column_dense)
        with self.assertRaises(ValueError):
            _maximum(self.column_dense, self.row_sparse)

        _maximum(self.single_flat_dense, self.single_flat_dense)
        _maximum(self.single_flat_dense, self.single_dense)
        _maximum(self.single_flat_dense, self.single_sparse)
        with self.assertRaises(ValueError):
            _maximum(self.single_flat_dense, self.row_flat_dense)

        _maximum(self.single_dense, self.single_flat_dense)
        _maximum(self.single_dense, self.single_dense)
        _maximum(self.single_dense, self.single_sparse)
        with self.assertRaises(ValueError):
            _maximum(self.single_flat_dense, self.row_flat_dense)

        _maximum(self.single_sparse, self.single_flat_dense)
        _maximum(self.single_sparse, self.single_dense)
        _maximum(self.single_sparse, self.single_sparse)
        with self.assertRaises(ValueError):
            _maximum(self.single_flat_dense, self.row_sparse)

        e_max = self.empty_dense.maximum(self.empty_dense)
        self.assertFalse((e_max != CArray([])).any())
        e_max = self.empty_dense.maximum(self.empty_sparse)
        self.assertFalse((e_max != CArray([])).any())
        e_max = self.empty_sparse.maximum(self.empty_dense)
        self.assertFalse((e_max != CArray([])).any())
        e_max = self.empty_sparse.maximum(self.empty_sparse)
        self.assertFalse((e_max != CArray([])).any())

    def test_minimum(self):
        """Test for CArray.minimum() method."""
        self.logger.info("Test for CArray.minimum() method.")

        def _minimum(array1, array2):
            array2 = array2.deepcopy()
            # Multiply each nonzero element of 2nd array to 2
            array2[array2 != 0] *= 2

            self.logger.info("a1: \n{:}".format(array1))
            self.logger.info("a2: \n{:}".format(array2))

            minimum_res = array1.minimum(array2)
            self.logger.info("a1.minimum(a2): \n{:}".format(minimum_res))

            if array1.ndim == 2 or array2.ndim == 2:
                array1 = array1.atleast_2d()
            self.assert_array_equal(minimum_res, array1)

        _minimum(self.array_sparse, self.array_sparse)
        _minimum(self.array_dense, self.array_dense)
        _minimum(self.array_sparse, self.array_dense)
        _minimum(self.array_dense, self.array_sparse)
        with self.assertRaises(ValueError):
            _minimum(self.array_sparse, self.array_sparse_sym)
        with self.assertRaises(ValueError):
            _minimum(self.array_dense, self.array_dense_sym)

        _minimum(self.row_flat_dense, self.row_flat_dense)
        _minimum(self.row_flat_dense, self.row_sparse)
        _minimum(self.row_flat_dense, self.row_dense)
        with self.assertRaises(ValueError):
            _minimum(self.row_flat_dense, self.column_dense)

        _minimum(self.row_sparse, self.row_sparse)
        _minimum(self.row_sparse, self.row_flat_dense)
        _minimum(self.row_sparse, self.row_dense)
        with self.assertRaises(ValueError):
            _minimum(self.row_sparse, self.column_sparse)

        _minimum(self.row_dense, self.row_dense)
        _minimum(self.row_dense, self.row_flat_dense)
        _minimum(self.row_dense, self.row_sparse)
        with self.assertRaises(ValueError):
            _minimum(self.row_dense, self.column_dense)

        _minimum(self.column_dense, self.column_dense)
        _minimum(self.column_dense, self.column_sparse)
        _minimum(self.column_sparse, self.column_sparse)
        _minimum(self.column_sparse, self.column_dense)
        with self.assertRaises(ValueError):
            _minimum(self.column_dense, self.row_sparse)

        _minimum(self.single_flat_dense, self.single_flat_dense)
        _minimum(self.single_flat_dense, self.single_dense)
        _minimum(self.single_flat_dense, self.single_sparse)
        with self.assertRaises(ValueError):
            _minimum(self.single_flat_dense, self.row_flat_dense)

        _minimum(self.single_dense, self.single_flat_dense)
        _minimum(self.single_dense, self.single_dense)
        _minimum(self.single_dense, self.single_sparse)
        with self.assertRaises(ValueError):
            _minimum(self.single_flat_dense, self.row_flat_dense)

        _minimum(self.single_sparse, self.single_flat_dense)
        _minimum(self.single_sparse, self.single_dense)
        _minimum(self.single_sparse, self.single_sparse)
        with self.assertRaises(ValueError):
            _minimum(self.single_flat_dense, self.row_sparse)

        e_min = self.empty_dense.minimum(self.empty_dense)
        self.assertFalse((e_min != CArray([])).any())
        e_min = self.empty_dense.minimum(self.empty_sparse)
        self.assertFalse((e_min != CArray([])).any())
        e_min = self.empty_sparse.minimum(self.empty_dense)
        self.assertFalse((e_min != CArray([])).any())
        e_min = self.empty_sparse.minimum(self.empty_sparse)
        self.assertFalse((e_min != CArray([])).any())


if __name__ == '__main__':
    CArrayTestCases.main()
