from secml.array.tests import CArrayTestCases

import numpy as np
import copy

from secml.array import CArray
from secml.core.constants import inf


class TestCArrayUtilsDataAlteration(CArrayTestCases):
    """Unit test for CArray UTILS - DATA ALTERATION methods."""

    def test_round(self):
        """Test for CArray.round() method."""
        self.logger.info("Test for CArray.round() method.")

        def _round(array):
            array_float = array.astype(float)
            array_float *= 1.0201
            self.logger.info("a: \n{:}".format(array))

            round_res = array_float.round()
            self.logger.info("a.round(): \n{:}".format(round_res))
            self.assert_array_equal(round_res, array.astype(float))

            round_res = array_float.round(decimals=2)
            self.logger.info("a.round(decimals=2): \n{:}".format(round_res))
            array_test = array * 1.02
            self.assert_array_equal(round_res, array_test)

            round_res = array_float.round(decimals=6)
            self.logger.info("a.round(decimals=6): \n{:}".format(round_res))
            self.assert_array_equal(round_res, array_float)

        _round(self.array_sparse)
        _round(self.row_sparse)
        _round(self.column_sparse)
        _round(self.array_dense)
        _round(self.row_sparse)
        _round(self.column_dense)

    def test_clip(self):
        """Test for CArray.clip() method."""
        self.logger.info("Test for CArray.clip() method.")

        def _check_clip(array, expected):
            self.logger.info("Array:\n{:}".format(array))

            intervals = [(0, 2), (0, inf), (-inf, 0)]

            for c_limits_idx, c_limits in enumerate(intervals):

                res = array.clip(*c_limits)
                self.logger.info("array.min(c_min={:}, c_max={:}):"
                                 "\n{:}".format(c_limits[0], c_limits[1], res))

                res_expected = expected[c_limits_idx]
                self.assertIsInstance(res, CArray)
                self.assertEqual(res.isdense, res_expected.isdense)
                self.assertEqual(res.issparse, res_expected.issparse)
                self.assertEqual(res.shape, res_expected.shape)
                self.assertEqual(res.dtype, res_expected.dtype)
                self.assertFalse((res != res_expected).any())

        # array_dense = CArray([[1, 0, 0, 5], [2, 4, 0, 0], [3, 6, 0, 0]]
        # row_flat_dense = CArray([4, 0, 6])

        _check_clip(self.array_dense,
                    (CArray([[1, 0, 0, 2], [2, 2, 0, 0], [2, 2, 0, 0]]),
                     CArray([[1., 0., 0., 5.],
                             [2., 4., 0., 0.],
                             [3., 6., 0., 0.]]),
                     CArray([[0., 0., 0., 0.],
                             [0., 0., 0., 0.],
                             [0., 0., 0., 0.]])))

        _check_clip(self.row_flat_dense, (CArray([2, 0, 2]),
                                          CArray([4., 0., 6.]),
                                          CArray([0., 0., 0.])))

        _check_clip(self.row_dense, (CArray([[2, 0, 2]]),
                                     CArray([[4., 0., 6.]]),
                                     CArray([[0., 0., 0.]])))

        _check_clip(self.column_dense, (CArray([[2], [0], [2]]),
                                        CArray([[4.], [0.], [6.]]),
                                        CArray([[0.], [0.], [0.]])))

        _check_clip(self.single_flat_dense,
                    (CArray([2]), CArray([4.]), CArray([0.])))
        _check_clip(self.single_dense,
                    (CArray([[2]]), CArray([[4.]]), CArray([[0.]])))

        # Check intervals wrongly chosen
        with self.assertRaises(ValueError):
            self.array_dense.clip(2, 0)
            self.array_dense.clip(0, -2)
            self.array_dense.clip(2, -2)

    def test_sort(self):
        """Test for CArray.sort() method."""
        self.logger.info("Test for CArray.sort() method")

        def _sort(axis, array, sorted_expected):
            self.logger.info("Array:\n{:}".format(array))

            array_isdense = array.isdense
            array_issparse = array.issparse

            for inplace in (False, True):
                array_copy = copy.deepcopy(array)
                array_sorted = array_copy.sort(axis=axis, inplace=inplace)
                self.logger.info("Array sorted along axis {:}:"
                                 "\n{:}".format(axis, array_sorted))

                self.assertEqual(array_issparse, array_sorted.issparse)
                self.assertEqual(array_isdense, array_sorted.isdense)

                self.assertFalse((sorted_expected != array_sorted).any())

                # Value we are going to replace to check inplace parameter
                alter_value = CArray(100, dtype=array_copy.dtype)

                if array_copy.ndim < 2:
                    array_copy[0] = alter_value
                    if inplace is False:
                        self.assertTrue(array_sorted[0] != alter_value[0])
                    else:
                        self.assertTrue(array_sorted[0] == alter_value[0])
                else:
                    array_copy[0, 0] = alter_value
                    if inplace is False:
                        self.assertTrue(array_sorted[0, 0] != alter_value[0])
                    else:
                        self.assertTrue(array_sorted[0, 0] == alter_value[0])

        # Sparse arrays
        _sort(-1, self.array_sparse,
              CArray([[0, 0, 1, 5], [0, 0, 2, 4], [0, 0, 3, 6]],
                     tosparse=True))
        _sort(0, self.array_sparse,
              CArray([[1, 0, 0, 0], [2, 4, 0, 0], [3, 6, 0, 5]],
                     tosparse=True))
        _sort(1, self.array_sparse,
              CArray([[0, 0, 1, 5], [0, 0, 2, 4], [0, 0, 3, 6]],
                     tosparse=True))

        _sort(-1, self.row_sparse, CArray([0, 4, 6], tosparse=True))
        _sort(0, self.row_sparse, CArray([4, 0, 6], tosparse=True))
        _sort(1, self.row_sparse, CArray([0, 4, 6], tosparse=True))

        _sort(-1, self.column_sparse, CArray([[4], [0], [6]], tosparse=True))
        _sort(0, self.column_sparse, CArray([[0], [4], [6]], tosparse=True))
        _sort(1, self.column_sparse, CArray([[4], [0], [6]], tosparse=True))

        # Dense arrays
        _sort(-1, self.array_dense,
              CArray([[0, 0, 1, 5], [0, 0, 2, 4], [0, 0, 3, 6]]))
        _sort(0, self.array_dense,
              CArray([[1, 0, 0, 0], [2, 4, 0, 0], [3, 6, 0, 5]]))
        _sort(1, self.array_dense,
              CArray([[0, 0, 1, 5], [0, 0, 2, 4], [0, 0, 3, 6]]))

        _sort(-1, self.row_dense, CArray([0, 4, 6]))
        _sort(0, self.row_dense, CArray([4, 0, 6]))
        _sort(1, self.row_dense, CArray([0, 4, 6]))

        _sort(-1, self.column_dense, CArray([[4], [0], [6]]))
        _sort(0, self.column_dense, CArray([[0], [4], [6]]))
        _sort(1, self.column_dense, CArray([[4], [0], [6]]))

        # Bool arrays
        _sort(-1, self.array_dense_bool,
              CArray([[False, True, True, True],
                      [False, False, False, False],
                      [True, True, True, True]]))
        _sort(0, self.array_dense_bool,
              CArray([[False, False, False, False],
                      [True, False, True, True],
                      [True, True, True, True]]))
        _sort(1, self.array_dense_bool,
              CArray([[False, True, True, True],
                      [False, False, False, False],
                      [True, True, True, True]]))

        _sort(-1, self.array_sparse_bool,
              CArray([[False, True, True, True],
                      [False, False, False, False],
                      [True, True, True, True]], tosparse=True))
        _sort(0, self.array_sparse_bool,
              CArray([[False, False, False, False],
                      [True, False, True, True],
                      [True, True, True, True]], tosparse=True))
        _sort(1, self.array_sparse_bool,
              CArray([[False, True, True, True],
                      [False, False, False, False],
                      [True, True, True, True]], tosparse=True))

        # Check sort() for empty arrays
        self.empty_flat_dense.sort()
        self.assertTrue((self.empty_flat_dense == CArray([])).all())
        self.empty_sparse.sort()
        self.assertTrue((self.empty_sparse == CArray([])).all())

    def test_argsort(self):
        """Test for CArray.argsort() method."""
        self.logger.info("Test for CArray.argsort() method.")

        def _argsort(axis, matrix):
            self.logger.info("array: {:}".format(matrix))
            sorted_idx = matrix.argsort(axis=axis)
            self.logger.info("array.argsort(axis={:}): {:}".format(axis, sorted_idx))

            self.assertFalse(sorted_idx.issparse, "sorted method don't return a cndarray")

            np_matrix = matrix.todense().tondarray()
            np_matrix = np.atleast_2d(np_matrix)
            np_sorted_idx = np.argsort(np_matrix, axis=axis)
            self.assertTrue((sorted_idx.tondarray() == np_sorted_idx).all())

        # Sparse arrays
        _argsort(None, self.array_sparse)
        _argsort(0, self.array_sparse)
        _argsort(1, self.array_sparse)

        _argsort(None, self.row_sparse)
        _argsort(0, self.row_sparse)
        _argsort(1, self.row_sparse)

        _argsort(None, self.column_sparse)
        _argsort(0, self.column_sparse)
        _argsort(1, self.column_sparse)

        # Dense arrays
        _argsort(None, self.array_dense)
        _argsort(0, self.array_dense)
        _argsort(1, self.array_dense)

        _argsort(None, self.row_dense)
        _argsort(0, self.row_dense)
        _argsort(1, self.row_dense)

        _argsort(None, self.column_dense)
        _argsort(0, self.column_dense)
        _argsort(1, self.column_dense)

        # Check argsort() for empty arrays
        sorted_matrix = CArray([], tosparse=False).argsort()
        self.assertTrue((sorted_matrix == CArray([])).all())
        sorted_matrix = CArray([], tosparse=True).argsort()
        self.assertTrue((sorted_matrix == CArray([])).all())

    def test_shuffle(self):
        """Test for CArray.shuffle() method."""
        self.logger.info("Test for CArray.shuffle() method")

        def _shuffle(array):

            array_copy = array.deepcopy()  # In-place method
            self.logger.info("Array:\n{:}".format(array))

            array_shape = array_copy.shape
            array_isdense = array_copy.isdense
            array_issparse = array_copy.issparse

            array.shuffle()
            self.logger.info("Array shuffled:\n{:}".format(array))

            self.assertEqual(array_shape, array.shape)
            self.assertEqual(array_issparse, array.issparse)
            self.assertEqual(array_isdense, array.isdense)

            array_list = array.tolist()
            array_copy_list = array_copy.tolist()
            # For vector-like arrays we shuffle the elements along axis 1
            if array.ndim == 2 and array.shape[0] == 1:
                array_list = array_list[0]
                array_copy_list = array_copy_list[0]
            for num in array_copy_list:
                # Return num position and remove it from the list
                # Not need of asserts here... .index() raises error
                # if num not in array_list
                array_list.pop(array_list.index(num))

        _shuffle(self.array_dense)
        _shuffle(self.array_sparse)

        _shuffle(self.row_sparse)
        _shuffle(self.column_sparse)

        _shuffle(self.row_flat_dense)
        _shuffle(self.row_dense)
        _shuffle(self.column_dense)

        _shuffle(self.array_dense_allzero)
        _shuffle(self.array_sparse_allzero)

        _shuffle(self.array_dense_bool)
        _shuffle(self.array_sparse_bool)

        _shuffle(self.empty_flat_dense)
        _shuffle(self.empty_sparse)
    

if __name__ == '__main__':
    CArrayTestCases.main()
