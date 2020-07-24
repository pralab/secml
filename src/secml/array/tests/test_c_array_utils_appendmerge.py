from secml.array.tests import CArrayTestCases

import numpy as np

from secml.array import CArray
from secml.core.type_utils import is_scalar


class TestCArrayUtilsAppendMerge(CArrayTestCases):
    """Unit test for CArray UTILS - APPEND/MERGE methods."""

    def test_append(self):
        """Test for CArray.append() method."""
        self.logger.info("Test for CArray.append() method.")

        def _append_allaxis(array1, array2):

            self.logger.info("a1: {:} ".format(array1))
            self.logger.info("a2: {:} ".format(array2))

            # default append, axis None (ravelled)
            append_res = array1.append(array2)
            self.logger.info("a1.append(a2): {:}".format(append_res))
            # If axis is None, result should be ravelled...
            if array1.isdense:
                self.assertEqual(1, append_res.ndim)
            else:  # ... but if array is sparse let's check for shape[0]
                self.assertEqual(1, append_res.shape[0])
            # Let's check the elements of the resulting array
            a1_comp = array1.todense().ravel()
            a2_comp = array2.todense().ravel()
            if array1.issparse:  # result will be sparse, so always 2d
                a1_comp = a1_comp.atleast_2d()
                a2_comp = a2_comp.atleast_2d()
            self.assert_array_equal(append_res[:array1.size], a1_comp)
            self.assert_array_equal(append_res[array1.size:], a2_comp)

            array1_shape0 = array1.atleast_2d().shape[0]
            array1_shape1 = array1.atleast_2d().shape[1]
            array2_shape0 = array2.atleast_2d().shape[0]
            array2_shape1 = array2.atleast_2d().shape[1]

            # check append on axis 0 (vertical)
            append_res = array1.append(array2, axis=0)
            self.logger.info("a1.append(a2, axis=0): {:}".format(append_res))
            self.assertEqual(array1_shape1, append_res.shape[1])
            self.assertEqual(
                array1_shape0 + array2_shape0, append_res.shape[0])
            self.assert_array_equal(append_res[array1_shape0:, :], array2)

            # check append on axis 1 (horizontal)
            append_res = array1.append(array2, axis=1)
            self.logger.info("a1.append(a2, axis=1): {:}".format(append_res))
            self.assertEqual(
                array1_shape1 + array2_shape1, append_res.shape[1])
            self.assertEqual(array1_shape0, append_res.shape[0])
            self.assert_array_equal(append_res[:, array1_shape1:], array2)

        _append_allaxis(self.array_dense, self.array_dense)
        _append_allaxis(self.array_sparse, self.array_sparse)
        _append_allaxis(self.array_sparse, self.array_dense)
        _append_allaxis(self.array_dense, self.array_sparse)

        # check append on empty arrays
        empty_sparse = CArray([], tosparse=True)
        empty_dense = CArray([], tosparse=False)
        self.assertTrue(
            (empty_sparse.append(empty_dense, axis=None) == empty_dense).all())
        self.assertTrue(
            (empty_sparse.append(empty_dense, axis=0) == empty_dense).all())
        self.assertTrue(
            (empty_sparse.append(empty_dense, axis=1) == empty_dense).all())

    def test_repeat(self):
        """Test for CArray.repeat() method."""
        self.logger.info("Test for CArray.repeat() method.")

        def _check_repeat(array):
            self.logger.info("Array:\n{:}".format(array))

            for axis in (None, 0, 1):

                if axis is None or array.ndim < 2:
                    repeats_add = CArray.randint(2, shape=array.size)
                elif axis == 0:
                    repeats_add = CArray.randint(2, shape=array.shape[0])
                elif axis == 1:
                    repeats_add = CArray.randint(2, shape=array.shape[1])
                else:
                    repeats_add = None

                for repeats in (0, 1, 2, repeats_add):

                    with self.assertRaises(TypeError):
                        array.repeat(repeats=np.array([1, 2]), axis=axis)

                    if axis == 1 and array.ndim < 2:
                        # No columns to repeat
                        with self.assertRaises(ValueError):
                            array.repeat(repeats=repeats, axis=axis)
                        continue

                    res = array.repeat(repeats=repeats, axis=axis)
                    self.logger.info("array.repeat({:}, axis={:}):"
                                     "\n{:}".format(repeats, axis, res))

                    self.assertIsInstance(res, CArray)
                    self.assertEqual(res.isdense, array.isdense)
                    self.assertEqual(res.issparse, array.issparse)
                    self.assertEqual(res.dtype, array.dtype)

                    if axis is None or array.ndim < 2:
                        # A flat array is always returned
                        if is_scalar(repeats):
                            repeats_mul = array.size * repeats
                        else:
                            repeats_mul = repeats.sum()
                        self.assertEqual(res.shape, (repeats_mul, ))
                    elif axis == 0:
                        if is_scalar(repeats):
                            repeats_mul = array.shape[0] * repeats
                        else:
                            repeats_mul = repeats.sum()
                        self.assertEqual(
                            res.shape, (repeats_mul, array.shape[1]))
                    elif axis == 1:
                        if is_scalar(repeats):
                            repeats_mul = array.shape[1] * repeats
                        else:
                            repeats_mul = repeats.sum()
                        self.assertEqual(
                            res.shape, (array.shape[0], repeats_mul))

                    if is_scalar(repeats):
                        repeats_size = array.size * repeats
                    else:
                        if axis is None or array.ndim < 2:
                            repeats_size = repeats.sum()
                        elif axis == 0:
                            repeats_size = repeats.sum() * array.shape[1]
                        elif axis == 1:
                            repeats_size = repeats.sum() * array.shape[0]
                        else:
                            repeats_size = None
                    self.assertEqual(res.size, repeats_size)

                    if not is_scalar(repeats):
                        repeats = repeats.tondarray()
                    np_res = array.tondarray().repeat(
                        repeats=repeats, axis=axis)
                    self.assertFalse((res.tondarray() != np_res).any())

        # array_dense = CArray([[1, 0, 0, 5], [2, 4, 0, 0], [3, 6, 0, 0]]
        # row_flat_dense = CArray([4, 0, 6])

        _check_repeat(self.array_dense)
        _check_repeat(self.row_flat_dense)
        _check_repeat(self.row_dense)
        _check_repeat(self.column_dense)
        _check_repeat(self.single_flat_dense)
        _check_repeat(self.single_dense)
        _check_repeat(self.array_dense_bool)
        _check_repeat(self.empty_dense)
        _check_repeat(self.empty_flat_dense)

        with self.assertRaises(NotImplementedError):
            self.array_sparse.repeat(2)

    def test_repmat(self):
        """Test for CArray.repmat() method."""
        self.logger.info("Test for CArray.repmat() method.")

        def _check_repmat(array_data):
            self.logger.info("array: {:}".format(array_data))

            rep_array = array_data.repmat(2, 3)
            self.logger.info("array.repmat(2, 3): {:}".format(rep_array))
            np_array = array_data.todense().tondarray()
            np_repeated_array = np.matlib.repmat(np_array, 2, 3)
            self.assertTrue((rep_array.tondarray() == np_repeated_array).all())

            rep_array = array_data.repmat(1, 3)
            self.logger.info("array.repmat(1, 3): {:}".format(rep_array))
            np_array = array_data.todense().tondarray()
            np_repeated_array = np.matlib.repmat(np_array, 1, 3)
            self.assertTrue((rep_array.tondarray() == np_repeated_array).all())

            rep_array = array_data.repmat(1, 2)
            self.logger.info("array.repmat(1, 2): {:}".format(rep_array))
            np_array = array_data.todense().tondarray()
            np_repeated_array = np.matlib.repmat(np_array, 1, 2)
            self.assertTrue((rep_array.tondarray() == np_repeated_array).all())

        for array in [self.row_flat_dense, self.row_sparse,
                      self.array_dense, self.array_sparse,
                      self.empty_sparse, self.empty_dense,
                      self.empty_flat_dense]:
            _check_repmat(array)


if __name__ == '__main__':
    CArrayTestCases.main()
