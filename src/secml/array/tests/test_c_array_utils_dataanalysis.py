from secml.array.tests import CArrayTestCases

import numpy as np

from secml.array import CArray
from secml.core.type_utils import is_scalar, is_int, is_list_of_lists
from secml.core.constants import nan, inf


class TestCArrayUtilsDataAnalysis(CArrayTestCases):
    """Unit test for CArray UTILS - DATA ANALYSIS methods."""

    def test_get_nnz(self):
        """Test for CArray.get_nnz()."""
        self.logger.info("Testing CArray.get_nnz() method")

        def check_nnz(array, expected):
            self.logger.info("array:\n{:}".format(array))
            for ax_i, ax in enumerate((None, 0, 1)):
                res = array.get_nnz(axis=ax)
                self.logger.info("get_nnz(axis={:}):\n{:}".format(ax, res))

                if ax is None:
                    self.assertIsInstance(res, int)
                    self.assertEqual(expected[ax_i], res)
                else:
                    self.assertIsInstance(res, CArray)
                    self.assertEqual(1, res.ndim)
                    if ax == 0:
                        self.assertEqual(array.shape[1], res.size)
                    elif ax == 1:
                        self.assertEqual(array.shape[0], res.size)
                    self.assertFalse((res != expected[ax_i]).any())

        check_nnz(self.array_sparse,
                  (6, CArray([3, 2, 0, 1]), CArray([2, 2, 2])))
        check_nnz(self.row_sparse,
                  (2, CArray([1, 0, 1]), CArray([2])))
        check_nnz(self.column_sparse,
                  (2, CArray([2]), CArray([1, 0, 1])))

        check_nnz(self.array_dense,
                  (6, CArray([3, 2, 0, 1]), CArray([2, 2, 2])))
        check_nnz(self.row_dense,
                  (2, CArray([1, 0, 1]), CArray([2])))
        check_nnz(self.column_dense,
                  (2, CArray([2]), CArray([1, 0, 1])))

        check_nnz(self.single_dense, (1, CArray([1]), CArray([1])))
        check_nnz(self.single_sparse, (1, CArray([1]), CArray([1])))

        # Empty arrays have shape (1, 0)
        check_nnz(self.empty_dense, (0, CArray([]), CArray([0])))
        check_nnz(self.empty_sparse, (0, CArray([]), CArray([0])))

    def test_unique(self):
        """Test for CArray.unique() method."""
        self.logger.info("Test for CArray.unique() method")

        def _unique(array, true_unique):

            self.logger.info("Array:\n{:}".format(array))

            if array.isdense:
                array_unique, u_indices, u_inverse, u_counts = array.unique(
                    return_index=True, return_inverse=True, return_counts=True)
                # Testing call without the optional parameters
                array_unique_single = array.unique()
            elif array.issparse:
                # return_inverse parameters are not available
                with self.assertRaises(NotImplementedError):
                    array.unique(return_inverse=True)
                array_unique, u_indices, u_counts = array.unique(
                    return_index=True, return_counts=True)
                # Testing call without the optional parameters
                array_unique_single = array.unique()
            else:
                raise ValueError("Unknown input array format")
            self.logger.info("array.unique():\n{:}".format(array_unique))

            self.assert_array_equal(array_unique, array_unique_single)

            self.assertIsInstance(array_unique, CArray)
            # output of unique method must be dense
            self.assertTrue(array_unique.isdense)

            self.assertEqual(true_unique.size, array_unique.size)

            unique_ok = True
            for num in true_unique:
                if num not in array_unique:
                    unique_ok = False
            self.assertTrue(unique_ok)

            # To make assert_array_equal work with sparse
            arr_comp = array_unique
            if array.issparse:
                arr_comp = arr_comp.atleast_2d()

            # unique_indices construct unique array from original FLAT one
            self.assertEqual(array_unique.size, u_indices.size)
            self.assertEqual(u_indices.dtype, int)
            self.assert_array_equal(array.ravel()[u_indices], arr_comp)

            self.assertEqual(array_unique.size, u_counts.size)
            self.assertEqual(u_counts.dtype, int)
            for e_idx, e in enumerate(array_unique):
                if e == 0:
                    # Comparing a sparse matrix with 0 using == is inefficient
                    e_num = array.size - (array != e).sum()
                else:
                    e_num = (array == e).sum()
                self.assertEqual(u_counts[e_idx].item(), e_num)

            if array.isdense:
                self.assertEqual(array.size, u_inverse.size)
                # unique_inverse reconstruct the original FLAT array
                self.assert_array_equal(array.ravel(), arr_comp[u_inverse])

        _unique(self.array_dense, CArray([0, 1, 2, 3, 4, 5, 6]))
        _unique(self.array_sparse, CArray([0, 1, 2, 3, 4, 5, 6]))

        _unique(self.row_sparse, CArray([4, 0, 6]))
        _unique(self.column_sparse, CArray([4, 0, 6]))

        _unique(self.row_flat_dense, CArray([4, 0, 6]))
        _unique(self.row_dense, CArray([4, 0, 6]))
        _unique(self.column_dense, CArray([4, 0, 6]))

        _unique(self.array_dense_allzero, CArray([0]))
        _unique(self.array_sparse_allzero, CArray([0]))

        _unique(self.array_dense_bool, CArray([True, False]))
        _unique(self.array_sparse_bool, CArray([True, False]))

        _unique(self.empty_flat_dense, CArray([]))
        _unique(self.empty_sparse, CArray([]))

    def test_bincount(self):
        """Test for CArray.bincount() method."""
        self.logger.info("Test for CArray.bincount() method.")

        def _check_bincount(array, expected, minlength=0):
            self.logger.info("Array:\n{:}".format(array))

            res = array.bincount(minlength=minlength)
            self.logger.info("array.bincount():\n{:}".format(res))

            self.assertTrue(res.is_vector_like)
            expected_length = array.max() + 1 if (minlength == 0) or (
                    minlength < array.max() + 1) \
                else minlength
            self.assertEqual(res.size, expected_length)
            self.assertFalse((res != expected).any())

        with self.assertRaises(ValueError):
            self.row_dense.bincount()
        with self.assertRaises(ValueError):
            self.array_dense.bincount()
        with self.assertRaises(ValueError):
            self.array_sparse.bincount()

        _check_bincount(self.row_sparse, CArray([1, 0, 0, 0, 1, 0, 1]))
        _check_bincount(self.row_flat_dense, CArray([1, 0, 0, 0, 1, 0, 1]))
        _check_bincount(self.single_flat_dense, CArray([0, 0, 0, 0, 1]))
        _check_bincount(self.single_bool_flat_dense, CArray([0, 1]))
        _check_bincount(self.row_flat_dense,
                        CArray([1, 0, 0, 0, 1, 0, 1, 0, 0, 0]), minlength=10)
        _check_bincount(self.row_sparse,
                        CArray([1, 0, 0, 0, 1, 0, 1, 0, 0, 0]), minlength=10)
        _check_bincount(
            self.row_sparse, CArray([1, 0, 0, 0, 1, 0, 1]), minlength=3)
        _check_bincount(self.row_flat_dense,
                        CArray([1, 0, 0, 0, 1, 0, 1]), minlength=3)

        # test when no zeros are present in input
        input_data = CArray([1, 2], tosparse=True)
        output_data = CArray([0, 1, 1])
        _check_bincount(input_data, output_data)
        _check_bincount(input_data, output_data, minlength=2)
        _check_bincount(input_data.todense().ravel(), output_data)
        _check_bincount(input_data.todense().ravel(), output_data, minlength=2)

        # test for negative minlength (exception is raised)
        with self.assertRaises(ValueError):
            self.row_flat_dense.bincount(minlength=-2)
        with self.assertRaises(ValueError):
            self.row_sparse.bincount(minlength=-2)

        # test vector with negative elements (exception is raised)
        with self.assertRaises(ValueError):
            (-self.row_flat_dense).bincount()
        with self.assertRaises(ValueError):
            (-self.row_sparse).bincount()

    def test_norm(self):
        """Test for CArray.norm() method."""
        self.logger.info("Test for CArray.norm() method.")

        # Few norms involve divisions by zeros in our test cases, that's ok
        self.logger.filterwarnings(
            action="ignore",
            message="divide by zero encountered in reciprocal",
            category=RuntimeWarning
        )
        self.logger.filterwarnings(
            action="ignore",
            message="divide by zero encountered in power",
            category=RuntimeWarning
        )

        def _check_norm(array):
            self.logger.info("array:\n{:}".format(array))

            for ord_idx, order in enumerate((None, 'fro', inf, -inf,
                                             0, 1, -1, 2, -2, 3, -3)):

                if order == 'fro':  # Frobenius is a matrix norm
                    self.logger.info(
                        "array.norm(order={:}): ValueError".format(order))
                    with self.assertRaises(ValueError):
                        array.norm(order=order)
                    continue

                # Scipy does not supports negative norms
                if array.issparse is True and is_int(order) and order < 0:
                    self.logger.info(
                        "array.norm(order={:}): ValueError".format(order))
                    with self.assertRaises(NotImplementedError):
                        array.norm(order=order)
                    continue

                res = array.norm(order=order)

                self.logger.info("array.norm(order={:}):\n{:}"
                                 "".format(order, res))

                # Special handle of empty arrays
                if array.size == 0:
                    self.assertTrue(is_scalar(res))
                    self.assertEqual(float, type(res))
                    self.assertEqual(0, res)
                    continue

                res_np = np.linalg.norm(
                    array.tondarray().ravel(), ord=order).round(4)

                res = round(res, 4)
                self.assertTrue(is_scalar(res))
                self.assertEqual(float, type(res))
                self.assertEqual(res_np, res)

        # Sparse arrays
        _check_norm(self.row_sparse)

        # Dense arrays
        _check_norm(self.row_flat_dense)
        _check_norm(self.row_dense)

        _check_norm(self.single_flat_dense)
        _check_norm(self.single_dense)
        _check_norm(self.single_sparse)

        _check_norm(self.empty_dense)
        _check_norm(self.empty_flat_dense)
        _check_norm(self.empty_sparse)

        # norm should be used on vector-like arrays only
        with self.assertRaises(ValueError):
            self.array_dense.norm()
        with self.assertRaises(ValueError):
            self.array_sparse.norm()

    def test_norm_2d(self):
        """Test for CArray.norm_2d() method."""
        self.logger.info("Test for CArray.norm_2d() method.")

        # Few norms involve divisions by zeros in our test cases, that's ok
        self.logger.filterwarnings(
            action="ignore",
            message="divide by zero encountered in reciprocal",
            category=RuntimeWarning
        )
        self.logger.filterwarnings(
            action="ignore",
            message="divide by zero encountered in power",
            category=RuntimeWarning
        )

        def _check_norm_2d(array):
            self.logger.info("array:\n{:}".format(array))

            for axis_idx, axis in enumerate((None, 0, 1)):
                for ord_idx, order in enumerate(
                        (None, 'fro', inf, -inf, 1, -1, 2, -2, 3, -3)):

                    if axis is None and order in (2, -2):
                        self.logger.info(
                            "array.norm_2d(order={:}, axis={:}): "
                            "NotImplementedError".format(order, axis))
                        # Norms not implemented for matrices
                        with self.assertRaises(NotImplementedError):
                            array.norm_2d(order=order, axis=axis)
                        continue

                    if axis is None and order in (3, -3):
                        self.logger.info(
                            "array.norm_2d(order={:}, axis={:}): "
                            "ValueError".format(order, axis))
                        # Invalid norm order for matrices
                        with self.assertRaises(ValueError):
                            array.norm_2d(order=order, axis=axis)
                        continue

                    if axis is not None and order == 'fro':
                        self.logger.info(
                            "array.norm_2d(order={:}, axis={:}): "
                            "ValueError".format(order, axis))
                        # fro-norm is a matrix norm
                        with self.assertRaises(ValueError):
                            array.norm_2d(order=order, axis=axis)
                        continue

                    if array.issparse is True and axis is not None and \
                            (is_int(order) and order < 0):
                        self.logger.info(
                            "array.norm_2d(order={:}, axis={:}): "
                            "NotImplementedError".format(order, axis))
                        # Negative vector norms not implemented for sparse
                        with self.assertRaises(NotImplementedError):
                            array.norm_2d(order=order, axis=axis)
                        continue

                    res = array.norm_2d(order=order, axis=axis)
                    self.logger.info("array.norm_2d(order={:}, axis={:}):"
                                     "\n{:}".format(order, axis, res))

                    # Special handle of empty arrays
                    if array.size == 0:
                        if axis is None:
                            self.assertTrue(is_scalar(res))
                            self.assertEqual(float, type(res))
                            self.assertEqual(0, res)
                        else:
                            self.assertEqual(2, res.ndim)  # Out always 2D
                            self.assertEqual(float, res.dtype)
                            self.assertFalse((CArray([[0.0]]) != res).any())
                        continue

                    res_np = np.linalg.norm(array.atleast_2d().tondarray(),
                                            ord=order, axis=axis,
                                            keepdims=True).round(4)

                    if axis is None:
                        res = round(res, 4)
                        self.assertTrue(is_scalar(res))
                        self.assertEqual(float, type(res))
                        self.assertEqual(res_np, res)
                    else:
                        res = res.round(4)
                        self.assertEqual(2, res.ndim)  # Out always 2D
                        if array.ndim == 1:
                            if axis == 0:  # Return a row
                                self.assertEqual(array.shape[0], res.shape[1])
                            else:  # Return an array with a single value
                                self.assertEqual(1, res.size)
                        else:
                            if axis == 0:  # Should return a row
                                self.assertEqual(1, res.shape[0])
                                self.assertEqual(
                                    array.shape[1], res.shape[1])
                            else:  # Should return a column
                                self.assertEqual(1, res.shape[1])
                                self.assertEqual(
                                    array.shape[0], res.shape[0])

                        self.assertEqual(res_np.dtype, res.dtype)

                        self.assertFalse((res_np != res.tondarray()).any())

                with self.assertRaises(ValueError):
                    self.logger.info("array.norm_2d(order={:}): "
                                     "NotImplementedError".format(0))
                    array.norm_2d(order=0)  # Norm 0 not implemented

        # Sparse arrays
        _check_norm_2d(self.array_sparse)
        _check_norm_2d(self.row_sparse)
        _check_norm_2d(self.column_sparse)

        # Dense arrays
        _check_norm_2d(self.array_dense)
        _check_norm_2d(self.row_flat_dense)
        _check_norm_2d(self.row_dense)
        _check_norm_2d(self.column_dense)

        # Bool arrays
        _check_norm_2d(self.array_dense_bool)
        _check_norm_2d(self.array_sparse_bool)

        _check_norm_2d(self.single_flat_dense)
        _check_norm_2d(self.single_dense)
        _check_norm_2d(self.single_sparse)

        _check_norm_2d(self.empty_dense)
        _check_norm_2d(self.empty_flat_dense)
        _check_norm_2d(self.empty_sparse)

    def test_sum(self):
        """Test for CArray.sum() method."""
        self.logger.info("Test for CArray.sum() method.")

        def _check_sum(array, expected):
            self.logger.info("Array:\n{:}".format(array))

            for keepdims in (True, False):
                for res_idx, axis in enumerate([None, 0, 1]):

                    res = array.sum(axis=axis, keepdims=keepdims)
                    self.logger.info("array.sum(axis={:}, keepdims={:}):"
                                     "\n{:}".format(axis, keepdims, res))

                    if axis is None:
                        self.assertTrue(is_scalar(res))
                    else:
                        self.assertIsInstance(res, CArray)

                    res_expected = expected[res_idx]
                    if not isinstance(res_expected, CArray):
                        self.assertIsInstance(res, type(res_expected))
                        self.assertEqual(res, res_expected)
                    else:
                        self.assertEqual(res.isdense, res_expected.isdense)
                        self.assertEqual(res.issparse, res_expected.issparse)
                        if keepdims is False:
                            res_expected = res_expected.ravel()
                        self.assertEqual(res.shape, res_expected.shape)
                        self.assertEqual(res.dtype, res_expected.dtype)
                        self.assertFalse((res != res_expected).any())

        # array_dense = CArray([[1, 0, 0, 5], [2, 4, 0, 0], [3, 6, 0, 0]]
        # row_flat_dense = CArray([4, 0, 6])

        self.logger.info("Testing CArray.sum()")

        _check_sum(self.array_sparse,
                   (21, CArray([[6, 10, 0, 5]]), CArray([[6], [6], [9]])))
        _check_sum(self.array_dense,
                   (21, CArray([[6, 10, 0, 5]]), CArray([[6], [6], [9]])))

        _check_sum(self.array_dense_bool,
                   (7, CArray([[2, 1, 2, 2]]), CArray([[3], [0], [4]])))
        _check_sum(self.array_sparse_bool,
                   (7, CArray([[2, 1, 2, 2]]), CArray([[3], [0], [4]])))

        _check_sum(self.row_flat_dense, (10, CArray([4, 0, 6]), CArray([10])))
        _check_sum(self.row_dense, (10, CArray([[4, 0, 6]]), CArray([[10]])))
        _check_sum(self.row_sparse, (10, CArray([[4, 0, 6]]), CArray([[10]])))

        _check_sum(self.column_dense,
                   (10, CArray([[10]]), CArray([[4], [0], [6]])))
        _check_sum(self.column_sparse,
                   (10, CArray([[10]]), CArray([[4], [0], [6]])))

        _check_sum(self.single_flat_dense, (4, CArray([4]), CArray([4])))
        _check_sum(self.single_dense, (4, CArray([[4]]), CArray([[4]])))
        _check_sum(self.single_sparse, (4, CArray([[4]]), CArray([[4]])))

        _check_sum(self.single_bool_flat_dense, (1, CArray([1]), CArray([1])))
        _check_sum(self.single_bool_dense, (1, CArray([[1]]), CArray([[1]])))
        _check_sum(self.single_bool_sparse, (1, CArray([[1]]), CArray([[1]])))

        _check_sum(self.empty_flat_dense, (0.0, CArray([0.0]), CArray([0.0])))
        _check_sum(self.empty_dense, (0.0, CArray([[0.0]]), CArray([[0.0]])))
        _check_sum(self.empty_sparse, (0.0, CArray([[0.0]]), CArray([[0.0]])))

    def test_cusum(self):
        """Test for CArray.cumsum() method."""
        self.logger.info("Test for CArray.cumsum() method.")

        def _check_cumsum(array):
            self.logger.info("Array:\n{:}".format(array))

            for res_idx, axis in enumerate([None, 0, 1]):
                for dtype in (None, float, int):

                    res = array.cumsum(axis=axis, dtype=dtype)
                    self.logger.info("array.cumsum(axis={:}, dtype={:}):"
                                     "\n{:}".format(axis, dtype, res))

                    res_np = np.cumsum(array.atleast_2d().tondarray(),
                                       axis=axis, dtype=dtype)

                    if array.ndim == 1:
                        # We pass to numpy 2D arrays but result
                        # for vectors will be flat
                        res_np = res_np.ravel()

                    self.assertIsInstance(res, CArray)
                    self.assertTrue(res.isdense)
                    self.assertEqual(res.shape, res_np.shape)
                    self.assertEqual(res.dtype, res_np.dtype)
                    self.assertFalse((res != CArray(res_np)).any())

        # array_dense = CArray([[1, 0, 0, 5], [2, 4, 0, 0], [3, 6, 0, 0]])
        # row_flat_dense = CArray([4, 0, 6])

        self.logger.info("Testing CArray.cumsum()")

        _check_cumsum(self.array_dense)

        _check_cumsum(self.row_flat_dense)
        _check_cumsum(self.row_dense)
        _check_cumsum(self.column_dense)

        _check_cumsum(self.single_flat_dense)
        _check_cumsum(self.single_dense)

        _check_cumsum(self.array_dense_bool)

        _check_cumsum(self.single_bool_flat_dense)
        _check_cumsum(self.single_bool_dense)

        _check_cumsum(self.empty_flat_dense)
        _check_cumsum(self.empty_dense)

        with self.assertRaises(NotImplementedError):
            _check_cumsum(self.array_sparse)

    def test_prod(self):
        """Test for CArray.prod() method."""
        self.logger.info("Test for CArray.prod() method.")

        def _check_prod(array, expected):
            self.logger.info("Array:\n{:}".format(array))

            for keepdims in (True, False):
                for dtype in [None, float, int]:
                    for res_idx, axis in enumerate([None, 0, 1]):

                        res = array.prod(
                            axis=axis, keepdims=keepdims, dtype=dtype)
                        self.logger.info(
                            "array.prod(axis={:}, keepdims={:}, dtype={:}):"
                            "\n{:}".format(axis, keepdims, dtype, res))

                        if axis is None:
                            self.assertTrue(is_scalar(res))
                        else:
                            self.assertIsInstance(res, CArray)

                        res_expected = expected[res_idx]
                        if not isinstance(res_expected, CArray):
                            if dtype is None:
                                if array.dtype.kind in ('i', 'u', 'b'):
                                    dtype_none = int
                                elif array.dtype.kind in ('f'):
                                    dtype_none = float
                                else:
                                    dtype_none = array.dtype.type
                                res_expected = dtype_none(res_expected)
                            else:
                                res_expected = dtype(res_expected)
                            self.assertIsInstance(res, type(res_expected))

                            self.assertEqual(res, res_expected)

                        else:
                            self.assertEqual(res.isdense,
                                             res_expected.isdense)
                            self.assertEqual(res.issparse,
                                             res_expected.issparse)

                            if keepdims is False:
                                res_expected = res_expected.ravel()
                            self.assertEqual(res.shape, res_expected.shape)

                            if dtype is None:
                                if array.dtype.kind in ('i', 'u', 'b'):
                                    dtype_none = int
                                elif array.dtype.kind in ('f'):
                                    dtype_none = float
                                else:
                                    dtype_none = array.dtype.type
                                res_expected = res_expected.astype(dtype_none)
                            else:
                                res_expected = res_expected.astype(dtype)
                            self.assertEqual(res.dtype, res_expected.dtype)

                            self.assertFalse((res != res_expected).any())

        # array_dense = CArray([[1, 0, 0, 5], [2, 4, 0, 0], [3, 6, 0, 0]]
        # row_flat_dense = CArray([4, 0, 6])

        _check_prod(self.array_sparse,
                    (0, CArray([[6, 0, 0, 0]], tosparse=True),
                     CArray([[0], [0], [0]], tosparse=True)))
        _check_prod(self.array_dense,
                    (0, CArray([[6, 0, 0, 0]]), CArray([[0], [0], [0]])))

        _check_prod(self.array_dense_bool,
                    (0, CArray([[0, 0, 0, 0]]), CArray([[0], [0], [1]])))
        _check_prod(self.array_sparse_bool,
                    (0, CArray([[0, 0, 0, 0]], tosparse=True),
                     CArray([[0], [0], [1]], tosparse=True)))

        _check_prod(self.row_flat_dense, (0, CArray([4, 0, 6]), CArray([0])))
        _check_prod(self.row_dense, (0, CArray([[4, 0, 6]]), CArray([[0]])))
        _check_prod(self.row_sparse, (0, CArray([[4, 0, 6]], tosparse=True),
                                      CArray([[0]], tosparse=True)))

        _check_prod(self.column_dense,
                    (0, CArray([[0]]), CArray([[4], [0], [6]])))
        _check_prod(self.column_sparse, (0, CArray([[0]], tosparse=True),
                                         CArray([[4], [0], [6]],
                                                tosparse=True)))

        _check_prod(self.single_flat_dense, (4, CArray([4]), CArray([4])))
        _check_prod(self.single_dense, (4, CArray([[4]]), CArray([[4]])))
        _check_prod(self.single_sparse, (4, CArray([[4]], tosparse=True),
                                         CArray([[4]], tosparse=True)))

        _check_prod(self.single_bool_flat_dense, (1, CArray([1]), CArray([1])))
        _check_prod(self.single_bool_dense, (1, CArray([[1]]), CArray([[1]])))
        _check_prod(self.single_bool_sparse, (1, CArray([[1]], tosparse=True),
                                              CArray([[1]], tosparse=True)))

        _check_prod(self.empty_flat_dense, (1.0, CArray([1.0]), CArray([1.0])))
        _check_prod(self.empty_dense, (1.0, CArray([[1.0]]), CArray([[1.0]])))
        _check_prod(self.empty_sparse, (1.0, CArray([[1.0]], tosparse=True),
                                        CArray([[1.0]], tosparse=True)))

    def test_all(self):
        """Test for CArray.all() method."""
        self.logger.info("Test for CArray.all() method")

        def _all(matrix, matrix_nozero, matrix_bool, matrix_bool_true):

            for axis in (None, 0, 1):

                if matrix.issparse is True and axis is not None:
                    with self.assertRaises(NotImplementedError):
                        matrix.all(axis=axis)
                else:
                    # all() on an array that contain also zeros gives False?
                    self.logger.info("matrix: \n" + str(matrix))
                    all_res = matrix.all(axis=axis)
                    self.logger.info("matrix.all(axis={:}) result is:\n"
                                     "{:}".format(axis, all_res))
                    if axis is None:
                        self.assertIsInstance(all_res, bool)
                        self.assertFalse(all_res)
                    else:
                        self.assertIsInstance(all_res, CArray)

                if matrix_nozero.issparse is True and axis is not None:
                    with self.assertRaises(NotImplementedError):
                        matrix_nozero.all(axis=axis)
                else:
                    # all() on an array with no zeros gives True?
                    self.logger.info("matrix_nozero: \n" + str(matrix_nozero))
                    all_res = matrix_nozero.all(axis=axis)
                    self.logger.info("matrix_nozero.all(axis={:}):\n"
                                     "{:}".format(axis, all_res))
                    if axis is None:
                        self.assertIsInstance(all_res, bool)
                        self.assertTrue(all_res)
                    else:
                        self.assertIsInstance(all_res, CArray)
                        self.assertFalse((all_res != True).any())

                if matrix_bool.issparse is True and axis is not None:
                    with self.assertRaises(NotImplementedError):
                        matrix_bool.all(axis=axis)
                else:
                    # all() on boolean array
                    self.logger.info("matrix_bool: \n" + str(matrix_bool))
                    all_res = matrix_bool.all(axis=axis)
                    self.logger.info("matrix_bool.all(axis={:}):\n"
                                     "{:}".format(axis, all_res))
                    if axis is None:
                        self.assertIsInstance(all_res, bool)
                        self.assertFalse(all_res)
                    else:
                        self.assertIsInstance(all_res, CArray)

                if matrix_bool_true.issparse is True and axis is not None:
                    with self.assertRaises(NotImplementedError):
                        matrix_bool_true.all(axis=axis)
                else:
                    # all() on a boolean array with all True
                    self.logger.info(
                        "matrix_bool_true: \n" + str(matrix_bool_true))
                    all_res = matrix_bool_true.all(axis=axis)
                    self.logger.info("matrix_bool_true.all(axis={:}):\n"
                                     "{:}".format(axis, all_res))
                    if axis is None:
                        self.assertIsInstance(all_res, bool)
                        self.assertTrue(all_res)
                    else:
                        self.assertIsInstance(all_res, CArray)
                        self.assertFalse((all_res != True).any())

        _all(self.array_sparse, self.array_sparse_nozero,
             self.array_sparse_bool, self.array_sparse_bool_true)
        _all(self.array_dense, self.array_dense_nozero,
             self.array_dense_bool, self.array_dense_bool_true)

    def test_any(self):
        """Test for CArray.any() method."""
        self.logger.info("Test for CArray.any() method")

        def _any(matrix, matrix_allzero, matrix_bool, matrix_bool_false):

            for axis in (None, 0, 1):

                if matrix.issparse is True and axis is not None:
                    with self.assertRaises(NotImplementedError):
                        matrix.any(axis=axis)
                else:
                    # any() on an array that contain also zeros gives True?
                    self.logger.info("matrix: \n" + str(matrix))
                    any_res = matrix.any(axis=axis)
                    self.logger.info("matrix.any(axis={:}):\n"
                                     "{:}".format(axis, any_res))
                    if axis is None:
                        self.assertIsInstance(any_res, bool)
                        self.assertTrue(any_res)
                    else:
                        self.assertIsInstance(any_res, CArray)

                if matrix_allzero.issparse is True and axis is not None:
                    with self.assertRaises(NotImplementedError):
                        matrix_allzero.any(axis=axis)
                else:
                    # any() on an array with all zeros gives False?
                    self.logger.info(
                        "matrix_allzero: \n" + str(matrix_allzero))
                    any_res = matrix_allzero.any(axis=axis)
                    self.logger.info("matrix_allzero.any(axis={:}):\n"
                                     "{:}".format(axis, any_res))
                    if axis is None:
                        self.assertIsInstance(any_res, bool)
                        self.assertFalse(any_res)
                    else:
                        self.assertIsInstance(any_res, CArray)
                        self.assertFalse((any_res != False).any())

                if matrix_bool.issparse is True and axis is not None:
                    with self.assertRaises(NotImplementedError):
                        matrix_bool.any(axis=axis)
                else:
                    # any() on boolean array
                    self.logger.info("matrix_bool: \n" + str(matrix_bool))
                    any_res = matrix_bool.any(axis=axis)
                    self.logger.info("matrix_bool.any(axis={:}):\n"
                                     "{:}".format(axis, any_res))
                    if axis is None:
                        self.assertIsInstance(any_res, bool)
                        self.assertTrue(any_res)
                    else:
                        self.assertIsInstance(any_res, CArray)

                if matrix_bool_false.issparse is True and axis is not None:
                    with self.assertRaises(NotImplementedError):
                        matrix_bool_false.any(axis=axis)
                else:
                    # any() on a boolean array with all False
                    self.logger.info(
                        "matrix_bool_false: \n" + str(matrix_bool_false))
                    any_res = matrix_bool_false.any(axis=axis)
                    self.logger.info("matrix_bool_false.any(axis={:}):\n"
                                     "{:}".format(axis, any_res))
                    if axis is None:
                        self.assertIsInstance(any_res, bool)
                        self.assertFalse(any_res)
                    else:
                        self.assertIsInstance(any_res, CArray)
                        self.assertFalse((any_res != False).any())

        _any(self.array_sparse, self.array_sparse_allzero,
             self.array_sparse_bool, self.array_sparse_bool_false)
        _any(self.array_dense, self.array_dense_allzero,
             self.array_dense_bool, self.array_dense_bool_false)

    def test_min_max_mean(self):
        """Test for CArray.min(), CArray.max(), CArray.mean() method."""
        self.logger.info(
            "Test for CArray.min(), CArray.max(), CArray.mean() method.")

        def _check_minmaxmean(func, array, expected):
            self.logger.info("Array:\n{:}".format(array))

            for keepdims in (True, False):
                for res_idx, axis in enumerate([None, 0, 1]):

                    if func == 'min':
                        res = array.min(axis=axis, keepdims=keepdims)
                        self.logger.info("array.min(axis={:}, keepdims={:}):"
                                         "\n{:}".format(axis, keepdims, res))
                    elif func == 'max':
                        res = array.max(axis=axis, keepdims=keepdims)
                        self.logger.info("array.max(axis={:}, keepdims={:}):"
                                         "\n{:}".format(axis, keepdims, res))
                    elif func == 'mean':
                        res = array.mean(axis=axis, keepdims=keepdims)
                        self.logger.info("array.mean(axis={:}, keepdims={:}):"
                                         "\n{:}".format(axis, keepdims, res))
                    else:
                        raise ValueError("func {:} unknown".format(func))

                    if axis is None:
                        self.assertTrue(is_scalar(res))
                    else:
                        self.assertIsInstance(res, CArray)

                    res_expected = expected[res_idx]
                    if not isinstance(res_expected, CArray):
                        res = CArray(res).round(2)[0]
                        self.assertEqual(res, res_expected)
                    else:
                        self.assertEqual(res.isdense, res_expected.isdense)
                        self.assertEqual(res.issparse, res_expected.issparse)
                        res = res.round(2)
                        if keepdims is False:
                            res_expected = res_expected.ravel()
                        self.assertEqual(res.shape, res_expected.shape)
                        self.assertFalse((res != res_expected).any())

        # array_dense = CArray([[1, 0, 0, 5], [2, 4, 0, 0], [3, 6, 0, 0]]
        # row_flat_dense = CArray([4, 0, 6])

        self.logger.info("Testing CArray.min()")

        _check_minmaxmean('min', self.array_sparse,
                          (0, CArray([[1, 0, 0, 0]]),
                           CArray([[0], [0], [0]])))
        _check_minmaxmean('min', self.array_dense,
                          (0, CArray([[1, 0, 0, 0]]), CArray([[0], [0], [0]])))

        _check_minmaxmean('min', self.row_flat_dense,
                          (0, CArray([4, 0, 6]), 0))
        _check_minmaxmean('min', self.row_sparse,
                          (0, CArray([[4, 0, 6]]),
                           CArray([[0]])))
        _check_minmaxmean('min', self.row_dense,
                          (0, CArray([[4, 0, 6]]), CArray([[0]])))

        _check_minmaxmean('min', self.column_sparse,
                          (0, CArray([[0]]),
                           CArray([[4], [0], [6]])))
        _check_minmaxmean('min', self.column_dense,
                          (0, CArray([[0]]), CArray([[4], [0], [6]])))

        _check_minmaxmean('min', self.single_flat_dense,
                          (4, CArray([4]), CArray([4])))
        _check_minmaxmean('min', self.single_dense,
                          (4, CArray([[4]]), CArray([[4]])))
        _check_minmaxmean('min', self.single_sparse,
                          (4, CArray([[4]]),
                           CArray([[4]])))

        self.logger.info("Testing CArray.max()")

        _check_minmaxmean('max', self.array_sparse,
                          (6, CArray([[3, 6, 0, 5]]),
                           CArray([[5], [4], [6]])))
        _check_minmaxmean('max', self.array_dense,
                          (6, CArray([[3, 6, 0, 5]]), CArray([[5], [4], [6]])))

        _check_minmaxmean('max', self.row_flat_dense,
                          (6, CArray([4, 0, 6]), CArray([6])))
        _check_minmaxmean('max', self.row_sparse,
                          (6, CArray([[4, 0, 6]]),
                           CArray([[6]])))
        _check_minmaxmean('max', self.row_dense,
                          (6, CArray([[4, 0, 6]]), CArray([[6]])))

        _check_minmaxmean('max', self.column_sparse,
                          (6, CArray([[6]]),
                           CArray([[4], [0], [6]])))
        _check_minmaxmean('max', self.column_dense,
                          (6, CArray([[6]]), CArray([[4], [0], [6]])))

        _check_minmaxmean('max', self.single_flat_dense,
                          (4, CArray([4]), CArray([4])))
        _check_minmaxmean('max', self.single_dense,
                          (4, CArray([[4]]), CArray([[4]])))
        _check_minmaxmean('max', self.single_sparse,
                          (4, CArray([[4]]),
                           CArray([[4]])))

        self.logger.info("Testing CArray.mean()")

        _check_minmaxmean('mean', self.array_sparse,
                          (1.75, CArray([[2, 3.33, 0, 1.67]]),
                           CArray([[1.5], [1.5], [2.25]])))
        _check_minmaxmean('mean', self.array_dense,
                          (1.75, CArray([[2, 3.33, 0, 1.67]]),
                           CArray([[1.5], [1.5], [2.25]])))

        _check_minmaxmean('mean', self.row_flat_dense,
                          (3.33, CArray([4, 0, 6]), CArray([3.33])))
        _check_minmaxmean('mean', self.row_sparse,
                          (3.33, CArray([[4, 0, 6]]), CArray([[3.33]])))
        _check_minmaxmean('mean', self.row_dense,
                          (3.33, CArray([[4, 0, 6]]), CArray([[3.33]])))

        _check_minmaxmean('mean', self.column_sparse,
                          (3.33, CArray([[3.33]]), CArray([[4], [0], [6]])))
        _check_minmaxmean('mean', self.column_dense,
                          (3.33, CArray([[3.33]]), CArray([[4], [0], [6]])))

        _check_minmaxmean('mean', self.single_flat_dense,
                          (4, CArray([4]), CArray([4])))
        _check_minmaxmean('mean', self.single_dense,
                          (4, CArray([[4]]), CArray([[4]])))
        _check_minmaxmean('mean', self.single_sparse,
                          (4, CArray([[4]]), CArray([[4]])))

    def test_nanmin_nanmax(self):
        """Test for CArray.nanmin(), CArray.nanmax() method."""
        self.logger.info(
            "Test for CArray.nanmin(), CArray.nanmax() method.")

        # We are going to test few cases when the results actually contain nans
        self.logger.filterwarnings(
            action="ignore",
            message="All-NaN slice encountered",
            category=RuntimeWarning
        )

        def _check_nanminnanmax(func, array, expected):
            # Adding few nans to array
            array = array.astype(float)  # Arrays with nans have float dtype
            array[0, 0] = nan

            self.logger.info("Array:\n{:}".format(array))

            for keepdims in (True, False):
                for res_idx, axis in enumerate([None, 0, 1]):

                    if func == 'nanmin':
                        res = array.nanmin(axis=axis, keepdims=keepdims)
                        self.logger.info(
                            "array.nanmin(axis={:}, keepdims={:}):"
                            "\n{:}".format(axis, keepdims, res))
                    elif func == 'nanmax':
                        res = array.nanmax(axis=axis, keepdims=keepdims)
                        self.logger.info(
                            "array.nanmax(axis={:}, keepdims={:}):"
                            "\n{:}".format(axis, keepdims, res))
                    else:
                        raise ValueError("func {:} unknown".format(func))

                    if axis is None:
                        self.assertTrue(is_scalar(res))
                    else:
                        self.assertIsInstance(res, CArray)

                    res_expected = expected[res_idx]
                    if not isinstance(res_expected, CArray):
                        res = CArray(res).round(2)[0]
                    else:
                        self.assertEqual(res.isdense, res_expected.isdense)
                        self.assertEqual(res.issparse,
                                         res_expected.issparse)
                        res = res.round(2)
                        if keepdims is False:
                            res_expected = res_expected.ravel()
                        self.assertEqual(res.shape, res_expected.shape)
                    # use numpy.testing to proper compare arrays with nans
                    self.assert_array_equal(res, res_expected)

        # array_dense = CArray([[1, 0, 0, 5], [2, 4, 0, 0], [3, 6, 0, 0]]
        # row_flat_dense = CArray([4, 0, 6])

        self.logger.info("Testing CArray.nanmin()")

        _check_nanminnanmax('nanmin', self.array_dense,
                            (0, CArray([[2, 0, 0, 0]]),
                             CArray([[0], [0], [0]])))

        _check_nanminnanmax('nanmin', self.row_flat_dense,
                            (0, CArray([nan, 0, 6]), CArray([0])))

        _check_nanminnanmax('nanmin', self.row_dense,
                            (0, CArray([[nan, 0, 6]]), CArray([[0]])))

        _check_nanminnanmax('nanmin', self.column_dense,
                            (0, CArray([[0]]), CArray([[nan], [0], [6]])))

        _check_nanminnanmax('nanmin', self.single_flat_dense,
                            (nan, CArray([nan]), CArray([nan])))
        _check_nanminnanmax('nanmin', self.single_dense,
                            (nan, CArray([[nan]]), CArray([[nan]])))

        self.logger.info("Testing CArray.nanmax()")

        _check_nanminnanmax('nanmax', self.array_dense,
                            (6, CArray([[3, 6, 0, 5]]),
                             CArray([[5], [4], [6]])))

        _check_nanminnanmax('nanmax', self.row_flat_dense,
                            (6, CArray([nan, 0, 6]), CArray([6])))

        _check_nanminnanmax('nanmax', self.row_dense,
                            (6, CArray([[nan, 0, 6]]), CArray([[6]])))

        _check_nanminnanmax('nanmax', self.column_dense,
                            (6, CArray([[6]]), CArray([[nan], [0], [6]])))

        _check_nanminnanmax('nanmax', self.single_flat_dense,
                            (nan, CArray([nan]), CArray([nan])))
        _check_nanminnanmax('nanmax', self.single_dense,
                            (nan, CArray([[nan]]), CArray([[nan]])))

        with self.assertRaises(NotImplementedError):
            self.array_sparse.nanmin()
        with self.assertRaises(NotImplementedError):
            self.array_sparse.nanmax()

    def test_argmin(self):
        """Test for CArray.argmin() method."""
        self.logger.info("Test for CArray.argmin() method.")

        def _argmin(array):
            self.logger.info("a: \n{:}".format(array))
            argmin_res = array.argmin(axis=None)
            self.logger.info("a.argmin(axis=None): \n{:}".format(argmin_res))
            self.assertIsInstance(argmin_res, int)
            min_res = array.min(axis=None)
            self.assertEqual(array.ravel()[argmin_res].item(), min_res)

            self.logger.info("a: \n{:}".format(array))
            argmin_res = array.argmin(axis=0)
            self.logger.info("a.argmin(axis=0): \n{:}".format(argmin_res))
            self.assertIsInstance(argmin_res, CArray)
            self.assertIsSubDtype(argmin_res.dtype, int)
            self.assertEqual(1, argmin_res.shape[0])
            # We create a find_2d-like mask to check result
            min_res = array.min(axis=0)
            argmin_res = [
                argmin_res.ravel().tolist(), list(range(array.shape[1]))]
            self.assert_array_equal(
                array[argmin_res].atleast_2d(), min_res)

            self.logger.info("a: \n{:}".format(array))
            argmin_res = array.argmin(axis=1)
            self.logger.info("a.argmin(axis=1): \n{:}".format(argmin_res))
            self.assertIsInstance(argmin_res, CArray)
            self.assertIsSubDtype(argmin_res.dtype, int)
            self.assertEqual(1, argmin_res.shape[1])
            # We create a find_2d-like mask to check result
            min_res = array.min(axis=1)
            min_res = min_res.T  # will return a column but we compare as a row
            argmin_res = [
                list(range(array.shape[0])), argmin_res.ravel().tolist()]
            self.assert_array_equal(
                array[argmin_res].atleast_2d(), min_res)

        _argmin(self.array_sparse)
        _argmin(self.row_sparse)
        _argmin(self.column_sparse)
        _argmin(self.array_dense)
        _argmin(self.row_sparse)
        _argmin(self.column_dense)

        # Repeat the test after converting to float
        _argmin(self.array_sparse.astype(float))
        _argmin(self.row_sparse.astype(float))
        _argmin(self.column_sparse.astype(float))
        _argmin(self.array_dense.astype(float))
        _argmin(self.row_sparse.astype(float))
        _argmin(self.column_dense.astype(float))

        # argmin on empty arrays should raise ValueError
        with self.assertRaises(ValueError):
            self.empty_flat_dense.argmin()
        with self.assertRaises(ValueError):
            self.empty_sparse.argmin()

    def test_argmax(self):
        """Test for CArray.argmax() method."""
        self.logger.info("Test for CArray.argmax() method.")

        def _argmax(array):
            self.logger.info("a: \n{:}".format(array))
            argmax_res = array.argmax(axis=None)
            self.logger.info("a.argmax(axis=None): \n{:}".format(argmax_res))
            self.assertIsInstance(argmax_res, int)
            max_res = array.max(axis=None)
            self.assertEqual(array.ravel()[argmax_res].item(), max_res)

            self.logger.info("a: \n{:}".format(array))
            argmax_res = array.argmax(axis=0)
            self.logger.info("a.argmax(axis=0): \n{:}".format(argmax_res))
            self.assertIsInstance(argmax_res, CArray)
            self.assertIsSubDtype(argmax_res.dtype, int)
            self.assertEqual(1, argmax_res.shape[0])
            # We create a find_2d-like mask to check result
            max_res = array.max(axis=0)
            argmax_res = [
                argmax_res.ravel().tolist(), list(range(array.shape[1]))]
            self.assert_array_equal(
                array[argmax_res].atleast_2d(), max_res)

            self.logger.info("a: \n{:}".format(array))
            argmax_res = array.argmax(axis=1)
            self.logger.info("a.argmax(axis=1): \n{:}".format(argmax_res))
            self.assertIsInstance(argmax_res, CArray)
            self.assertIsSubDtype(argmax_res.dtype, int)
            self.assertEqual(1, argmax_res.shape[1])
            # We create a find_2d-like mask to check result
            max_res = array.max(axis=1)
            max_res = max_res.T  # max return a column but we compare as a row
            argmax_res = [
                list(range(array.shape[0])), argmax_res.ravel().tolist()]
            self.assert_array_equal(
                array[argmax_res].atleast_2d(), max_res)

        _argmax(self.array_sparse)
        _argmax(self.row_sparse)
        _argmax(self.column_sparse)
        _argmax(self.array_dense)
        _argmax(self.row_sparse)
        _argmax(self.column_dense)

        # Repeat the test after converting to float
        _argmax(self.array_sparse.astype(float))
        _argmax(self.row_sparse.astype(float))
        _argmax(self.column_sparse.astype(float))
        _argmax(self.array_dense.astype(float))
        _argmax(self.row_sparse.astype(float))
        _argmax(self.column_dense.astype(float))

        # argmax on empty arrays should raise ValueError
        with self.assertRaises(ValueError):
            self.empty_flat_dense.argmax()
        with self.assertRaises(ValueError):
            self.empty_sparse.argmax()

    def test_nanargmin(self):
        """Test for CArray.nanargmin() method."""
        self.logger.info("Test for CArray.nanargmin() method.")

        def _check_nanargmin(array):
            # NOTE: WE use numpy.testing to proper compare arrays with nans

            # Adding few nans to array
            array = array.astype(float)  # Arrays with nans have float dtype
            array[0, 0] = nan

            self.logger.info("a: \n{:}".format(array))
            argmin_res = array.nanargmin(axis=None)
            self.logger.info(
                "a.nanargmin(axis=None): \n{:}".format(argmin_res))
            self.assertIsInstance(argmin_res, int)
            min_res = array.nanmin(axis=None)
            # use numpy.testing to proper compare arrays with nans
            self.assert_array_equal(array.ravel()[argmin_res], min_res)

            self.logger.info("a: \n{:}".format(array))
            if array.shape[0] == 1:
                # ValueError: All-NaN slice encountered
                with self.assertRaises(ValueError):
                    array.nanargmin(axis=0)
            else:
                argmin_res = array.nanargmin(axis=0)
                self.logger.info(
                    "a.nanargmin(axis=0): \n{:}".format(argmin_res))
                self.assertIsInstance(argmin_res, CArray)
                min_res = array.nanmin(axis=0)
                # One res for each column with keepdims
                min_res = min_res.ravel()
                argmin_res = [
                    argmin_res.ravel().tolist(), list(range(array.shape[1]))]
                # use numpy.testing to proper compare arrays with nans
                self.assert_array_equal(array[argmin_res], min_res)

            self.logger.info("a: \n{:}".format(array))
            if array.shape[1] == 1:
                # ValueError: All-NaN slice encountered
                with self.assertRaises(ValueError):
                    array.nanargmin(axis=1)
            else:
                argmin_res = array.nanargmin(axis=1)
                self.assertIsInstance(argmin_res, CArray)
                self.logger.info(
                    "a.nanargmin(axis=1): \n{:}".format(argmin_res))
                min_res = array.nanmin(axis=1)
                # One res for each row with keepdims
                min_res = min_res.ravel()
                argmin_res = [
                    list(range(array.shape[0])), argmin_res.ravel().tolist()]
                # use numpy.testing to proper compare arrays with nans
                self.assert_array_equal(array[argmin_res], min_res)

        _check_nanargmin(self.array_dense)
        _check_nanargmin(self.row_dense)
        _check_nanargmin(self.column_dense)

        # nanargmin on empty arrays should raise ValueError
        with self.assertRaises(ValueError):
            self.empty_dense.nanargmin()

        with self.assertRaises(NotImplementedError):
            self.array_sparse.nanargmin()

    def test_nanargmax(self):
        """Test for CArray.nanargmax() method."""
        self.logger.info("Test for CArray.nanargmax() method.")

        def _check_nanargmax(array):
            # NOTE: WE use numpy.testing to proper compare arrays with nans

            # Adding few nans to array
            array = array.astype(float)  # Arrays with nans have float dtype
            array[0, 0] = nan

            self.logger.info("a: \n{:}".format(array))
            argmax_res = array.nanargmax(axis=None)
            self.logger.info(
                "a.nanargmax(axis=None): \n{:}".format(argmax_res))
            self.assertIsInstance(argmax_res, int)
            max_res = array.nanmax(axis=None)
            self.assert_array_equal(array.ravel()[argmax_res], max_res)

            self.logger.info("a: \n{:}".format(array))
            if array.shape[0] == 1:
                # ValueError: All-NaN slice encountered
                with self.assertRaises(ValueError):
                    array.nanargmax(axis=0)
            else:
                argmax_res = array.nanargmax(axis=0)
                self.logger.info(
                    "a.nanargmax(axis=0): \n{:}".format(argmax_res))
                self.assertIsInstance(argmax_res, CArray)
                max_res = array.nanmax(axis=0)
                # One res for each column with keepdims
                max_res = max_res.ravel()
                argmax_res = [
                    argmax_res.ravel().tolist(), list(range(array.shape[1]))]
                self.assert_array_equal(array[argmax_res], max_res)

            self.logger.info("a: \n{:}".format(array))
            if array.shape[1] == 1:
                # ValueError: All-NaN slice encountered
                with self.assertRaises(ValueError):
                    array.nanargmax(axis=1)
            else:
                argmax_res = array.nanargmax(axis=1)
                self.logger.info(
                    "a.nanargmax(axis=1): \n{:}".format(argmax_res))
                self.assertIsInstance(argmax_res, CArray)
                max_res = array.nanmax(axis=1)
                # One res for each row with keepdims
                max_res = max_res.ravel()
                argmax_res = [
                    list(range(array.shape[0])), argmax_res.ravel().tolist()]
                self.assert_array_equal(array[argmax_res], max_res)

        _check_nanargmax(self.array_dense)
        _check_nanargmax(self.row_dense)
        _check_nanargmax(self.column_dense)

        # nanargmax on empty arrays should raise ValueError
        with self.assertRaises(ValueError):
            self.empty_dense.nanargmax()

        with self.assertRaises(NotImplementedError):
            self.array_sparse.nanargmax()

    def test_median(self):
        """Test for CArray.median() method."""
        self.logger.info("Test for CArray.median() method.")

        def _check_median(array, expected):
            self.logger.info("Array:\n{:}".format(array))

            for keepdims in (True, False):
                for res_idx, axis in enumerate([None, 0, 1]):

                    res = array.median(axis=axis, keepdims=keepdims)
                    self.logger.info(
                        "array.median(axis={:}, keepdims={:}):"
                        "\n{:}".format(axis, keepdims, res))

                    if axis is None:
                        self.assertTrue(is_scalar(res))
                    else:
                        self.assertIsInstance(res, CArray)

                    res_expected = expected[res_idx]
                    if not isinstance(res_expected, CArray):
                        res = CArray(res).round(2)[0]
                        self.assertEqual(res, res_expected)
                    else:
                        self.assertEqual(res.isdense, res_expected.isdense)
                        self.assertEqual(res.issparse,
                                         res_expected.issparse)
                        res = res.round(2)
                        if keepdims is False:
                            res_expected = res_expected.ravel()
                        self.assertEqual(res.shape, res_expected.shape)
                        self.assertFalse((res != res_expected).any())

        # array_dense = CArray([[1, 0, 0, 5], [2, 4, 0, 0], [3, 6, 0, 0]]
        # row_flat_dense = CArray([4, 0, 6])

        _check_median(self.array_dense, (0.5, CArray([[2, 4.0, 0, 0.]]),
                                         CArray([[0.5], [1.], [1.5]])))

        _check_median(self.row_flat_dense,
                      (4.0, CArray([4, 0, 6]), CArray([4.0])))

        _check_median(self.row_dense,
                      (4.0, CArray([[4, 0, 6]]), CArray([[4.0]])))

        _check_median(self.column_dense,
                      (4.0, CArray([[4.0]]), CArray([[4], [0], [6]])))

        _check_median(self.single_flat_dense, (4, CArray([4]), CArray([4])))
        _check_median(self.single_dense, (4, CArray([[4]]), CArray([[4]])))

        with self.assertRaises(NotImplementedError):
            self.array_sparse.median()

    def test_sha1(self):
        """Test for CArray.sha1() method."""
        self.logger.info("Test for CArray.sha1() method.")

        def _check_sha1(array):
            self.logger.info("Array:\n{:}".format(array))

            sha1 = array.sha1()
            self.logger.info("array.sha1():\n{:}".format(sha1))

            self.assertIsInstance(sha1, str)

            # Transpose the array and check if sha1 changes if shape changes
            array_mod = array.T
            self.logger.info(
                "Checking hash after transpose:\n{:}".format(array_mod))
            sha1_mod = array_mod.sha1()
            self.logger.info("array_mod.sha1():\n{:}".format(sha1_mod))
            if array_mod.shape != array.shape:
                self.assertNotEqual(sha1, sha1_mod)
            else:  # If shape didn't change (empty or single elem array)
                self.assertEqual(sha1, sha1_mod)

            # Change dtype and check if sha1 changes if data changes
            newtype = int if array.dtype != int else float
            array_mod = array.astype(newtype)
            self.logger.info("Checking hash after changing dtype to "
                             "{:}:\n{:}".format(newtype, array_mod))
            sha1_mod = array_mod.sha1()
            self.logger.info("array_mod.sha1():\n{:}".format(sha1_mod))
            if array_mod.size > 0:
                self.assertNotEqual(sha1, sha1_mod)
            else:  # Empty array, no data to change. has should be the same
                self.assertEqual(sha1, sha1_mod)

            return sha1

        sha1_list = [
            _check_sha1(self.array_sparse),
            _check_sha1(self.array_dense),
            _check_sha1(self.array_dense_bool),
            _check_sha1(self.array_sparse_bool),
            _check_sha1(self.row_flat_dense),
            _check_sha1(self.row_dense),
            _check_sha1(self.row_sparse),
            _check_sha1(self.column_dense),
            _check_sha1(self.column_sparse),
            _check_sha1(self.single_flat_dense),
            _check_sha1(self.single_dense),
            _check_sha1(self.single_sparse),
            _check_sha1(self.single_bool_flat_dense),
            _check_sha1(self.single_bool_dense),
            _check_sha1(self.single_bool_sparse),
            _check_sha1(self.empty_flat_dense),
            _check_sha1(self.empty_dense),
            _check_sha1(self.empty_sparse)
        ]

        # We now check that all the collected hashes are different
        # as each test case was different
        import itertools
        for a, b in itertools.combinations(sha1_list, 2):
            self.assertNotEqual(a, b)

    def test_is_inf_nan(self):
        """Test for CArray .is_inf, .is_posinf, .is_neginf, .is_nan methods."""

        def _check_is_inf_nan(fun, val, array, pos=None):

            if pos is not None:
                array = array.astype(float)  # To correctly assign inf/nan
                array[pos] = val

            self.logger.info("Array:\n{:}".format(array))

            res = fun(array)
            self.logger.info("array.{:}():\n{:}".format(fun.__name__, res))

            self.assertIsInstance(res, CArray)
            self.assertIsSubDtype(res.dtype, bool)
            self.assertEqual(array.issparse, res.issparse)
            self.assertEqual(array.shape, res.shape)

            if pos is not None:
                self.assertTrue(all(res[pos]))
                self.assertEqual(
                    len(pos[0]) if is_list_of_lists(pos) else len(pos),
                    res.nnz)

        for test_fun, sub_val in (
                (CArray.is_inf, inf), (CArray.is_inf, -inf),
                (CArray.is_posinf, inf), (CArray.is_neginf, -inf),
                (CArray.is_nan, nan)):
            self.logger.info(
                "Test for CArray.{:}() method.".format(test_fun.__name__))

            _check_is_inf_nan(
                test_fun, sub_val, self.array_sparse, [[0, 1], [1, 2]]),
            _check_is_inf_nan(
                test_fun, sub_val, self.array_dense, [[0, 1], [1, 2]]),
            _check_is_inf_nan(
                test_fun, sub_val, self.array_dense_bool, [[0, 1], [1, 2]]),
            _check_is_inf_nan(
                test_fun, sub_val, self.array_sparse_bool, [[0, 1], [1, 2]]),
            _check_is_inf_nan(
                test_fun, sub_val, self.row_flat_dense, [1, 2]),
            _check_is_inf_nan(
                test_fun, sub_val, self.row_dense, [1, 2]),
            _check_is_inf_nan(
                test_fun, sub_val, self.row_sparse, [1, 2]),
            _check_is_inf_nan(
                test_fun, sub_val, self.column_dense, [[1, 2], [0, 0]]),
            _check_is_inf_nan(
                test_fun, sub_val, self.column_sparse, [[1, 2], [0, 0]]),
            _check_is_inf_nan(
                test_fun, sub_val, self.single_flat_dense, [0]),
            _check_is_inf_nan(
                test_fun, sub_val, self.single_dense, [0]),
            _check_is_inf_nan(
                test_fun, sub_val, self.single_sparse, [0]),
            _check_is_inf_nan(
                test_fun, sub_val, self.single_bool_flat_dense, [0]),
            _check_is_inf_nan(
                test_fun, sub_val, self.single_bool_dense, [0]),
            _check_is_inf_nan(
                test_fun, sub_val, self.single_bool_sparse, [0]),
            _check_is_inf_nan(
                test_fun, sub_val, self.empty_flat_dense),
            _check_is_inf_nan(
                test_fun, sub_val, self.empty_dense),
            _check_is_inf_nan(
                test_fun, sub_val, self.empty_sparse)


if __name__ == '__main__':
    CArrayTestCases.main()
