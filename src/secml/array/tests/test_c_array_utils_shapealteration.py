from secml.array.tests import CArrayTestCases

import numpy as np

from secml.array import CArray


class TestCArrayUtilsShapeAlteration(CArrayTestCases):
    """Unit test for CArray UTILS : SHAPE ALTERATION methods."""

    def test_transpose(self):
        """Method that test array transposing."""
        target_sparse = self.array_sparse.get_data().transpose()
        target_dense = self.array_dense.get_data().transpose()

        self.assertFalse(
            (target_sparse != self.array_sparse.transpose().get_data()).todense().any())
        self.assertFalse(
            (target_dense != self.array_dense.transpose().get_data()).any())

        target_sparse = self.array_sparse.get_data().T
        target_dense = self.array_dense.get_data().T

        self.assertFalse(
            (target_sparse != self.array_sparse.transpose().get_data()).todense().any())
        self.assertFalse(
            (target_dense != self.array_dense.transpose().get_data()).any())

        dense_flat = CArray([1, 2, 3])
        self.logger.info("We have a flat vector {:}".format(dense_flat))
        dense_flat_transposed = dense_flat.T
        self.logger.info("We transposed the vector: {:}. Shape {:}"
                         "".format(dense_flat_transposed,
                                   dense_flat_transposed.shape))
        self.assertEqual(len(dense_flat_transposed.shape), 2,
                         "Array still flat after transposing!")

    def test_ravel(self):
        """Test for CArray.ravel() method."""
        self.logger.info("Test for CArray.ravel() method.")

        def _check_ravel(array):
            self.logger.info("array:\n{:}".format(array))

            array_ravel = array.ravel()
            self.logger.info("array.ravel():\n{:}".format(array_ravel))

            self.assertIsInstance(array_ravel, CArray)

            self.assertEqual(array.size, array_ravel.size)

            if array.isdense:
                self.assertEqual(array_ravel.ndim, 1)
                self.assertEqual(array_ravel.shape, (array.size, ))
            if array.issparse:
                self.assertEqual(array_ravel.shape[0], 1)
                self.assertEqual(array_ravel.shape, (1, array.size))

        _check_ravel(self.array_dense)
        _check_ravel(self.array_sparse)

        _check_ravel(self.array_dense_bool)
        _check_ravel(self.array_sparse_bool)

        _check_ravel(self.row_flat_dense)
        _check_ravel(self.row_dense)
        _check_ravel(self.column_dense)

        _check_ravel(self.row_sparse)
        _check_ravel(self.column_sparse)

        _check_ravel(self.single_flat_dense)
        _check_ravel(self.single_dense)
        _check_ravel(self.single_sparse)

        _check_ravel(self.single_bool_flat_dense)
        _check_ravel(self.single_bool_dense)
        _check_ravel(self.single_bool_sparse)

        _check_ravel(self.empty_dense)
        _check_ravel(self.empty_flat_dense)
        _check_ravel(self.empty_sparse)

    def test_flatten(self):
        """Test for CArray.flatten() method."""
        self.logger.info("Test for CArray.flatten() method.")

        def _check_flatten(array):
            self.logger.info("array:\n{:}".format(array))

            array_original = array.deepcopy()

            array_flatten = array.flatten()
            self.logger.info("array.flatten():\n{:}".format(array_flatten))

            if isinstance(array_flatten, CArray):

                self.assertEqual(array.size, array_flatten.size)

                if array.isdense:
                    self.assertEqual(array_flatten.ndim, 1)
                if array.issparse:
                    self.assertEqual(array_flatten.shape[0], 1)

                array_flatten *= 5

                # Check if .flatten() made a copy
                self.assertFalse((array_original != array).any())

            else:
                self.assertIsInstance(array_flatten,
                                      (bool, float, np.bool_, int, np.integer))

        _check_flatten(self.array_dense)
        _check_flatten(self.array_sparse)

        _check_flatten(self.array_dense_bool)
        _check_flatten(self.array_sparse_bool)

        _check_flatten(self.row_flat_dense)
        _check_flatten(self.row_dense)
        _check_flatten(self.column_dense)

        _check_flatten(self.row_sparse)
        _check_flatten(self.column_sparse)

        _check_flatten(self.single_flat_dense)
        _check_flatten(self.single_dense)
        _check_flatten(self.single_sparse)

        _check_flatten(self.single_bool_flat_dense)
        _check_flatten(self.single_bool_dense)
        _check_flatten(self.single_bool_sparse)

        _check_flatten(self.empty_dense)
        _check_flatten(self.empty_flat_dense)
        _check_flatten(self.empty_sparse)

    def test_reshape(self):
        """Test for CArray.test_reshape() method."""
        self.logger.info("Test for CArray.test_reshape() method.")

        def _check_reshape(array, shape):
            self.logger.info("Array:\n{:}".format(array))

            res = array.reshape(newshape=shape)
            self.logger.info(
                "array.reshape(newshape={:}):\n{:}".format(shape, res))

            # Transforming input shape to appropriate tuple
            if not isinstance(shape, tuple):
                shape = (shape, )
            if array.issparse:
                if len(shape) == 1:
                    shape = (1, shape[0])

            self.assertEqual(res.dtype, array.dtype)
            self.assertEqual(res.shape, shape)

            # We now go to check if original array elements are preserved
            self.assertFalse(
                (CArray(array.ravel()) != CArray(res.ravel())).any())

        _check_reshape(self.array_dense, (2, 6))
        _check_reshape(self.array_dense, 12)
        with self.assertRaises(ValueError):
            _check_reshape(self.array_dense, (2, 4))
        _check_reshape(self.array_sparse, (2, 6))
        _check_reshape(self.array_sparse, 12)
        with self.assertRaises(ValueError):
            _check_reshape(self.array_sparse, (2, 4))

        _check_reshape(self.array_dense_bool, (2, 6))
        _check_reshape(self.array_dense_bool, 12)
        with self.assertRaises(ValueError):
            _check_reshape(self.array_dense_bool, (2, 4))
        _check_reshape(self.array_sparse_bool, (2, 6))
        _check_reshape(self.array_sparse_bool, 12)
        with self.assertRaises(ValueError):
            _check_reshape(self.array_sparse_bool, (2, 4))

        _check_reshape(self.row_flat_dense, (3, 1))
        _check_reshape(self.row_flat_dense, (1, 3))
        _check_reshape(self.row_flat_dense, 3)
        with self.assertRaises(ValueError):
            _check_reshape(self.row_flat_dense, (2, 4))
        _check_reshape(self.row_dense, (3, 1))
        _check_reshape(self.row_dense, 3)
        with self.assertRaises(ValueError):
            _check_reshape(self.row_dense, (2, 4))

        _check_reshape(self.column_dense, (1, 3))
        _check_reshape(self.column_dense, 3)
        with self.assertRaises(ValueError):
            _check_reshape(self.column_dense, (2, 4))
        _check_reshape(self.column_sparse, (1, 3))
        _check_reshape(self.column_sparse, 3)
        with self.assertRaises(ValueError):
            _check_reshape(self.column_sparse, (2, 4))

        _check_reshape(self.single_flat_dense, (1, 1))
        with self.assertRaises(ValueError):
            _check_reshape(self.single_flat_dense, (2, 4))
        _check_reshape(self.single_dense, 1)
        with self.assertRaises(ValueError):
            _check_reshape(self.single_dense, (2, 4))
        _check_reshape(self.single_sparse, 1)
        with self.assertRaises(ValueError):
            _check_reshape(self.single_sparse, (2, 4))

        _check_reshape(self.single_bool_flat_dense, (1, 1))
        with self.assertRaises(ValueError):
            _check_reshape(self.single_bool_flat_dense, (2, 4))
        _check_reshape(self.single_bool_dense, 1)
        with self.assertRaises(ValueError):
            _check_reshape(self.single_bool_dense, (2, 4))
        _check_reshape(self.single_bool_sparse, 1)
        with self.assertRaises(ValueError):
            _check_reshape(self.single_bool_sparse, (2, 4))

        with self.assertRaises(ValueError):
            _check_reshape(self.empty_flat_dense, (1, 1))
        with self.assertRaises(ValueError):
            _check_reshape(self.empty_flat_dense, 1)
        with self.assertRaises(ValueError):
            _check_reshape(self.empty_dense, (1, 1))
        with self.assertRaises(ValueError):
            _check_reshape(self.empty_dense, 1)
        with self.assertRaises(ValueError):
            _check_reshape(self.empty_sparse, (1, 1))
        with self.assertRaises(ValueError):
            _check_reshape(self.empty_sparse, 1)

    def test_resize(self):
        """Test for CArray.resize() method."""
        self.logger.info("Test for CArray.resize() method.")

        def _check_resize(array, shape):
            self.logger.info("Array:\n{:}".format(array))

            for constant in [0, 2, True, False]:
                res = array.resize(newshape=shape, constant=constant)
                self.logger.info(
                    "array.resize(newshape={:}, constant={:}):"
                    "\n{:}".format(shape, constant, res))

                if not isinstance(shape, tuple):
                    self.assertEqual(res.ndim, 1)
                    self.assertEqual(res.size, shape)
                else:
                    self.assertEqual(res.shape, shape)
                self.assertEqual(res.dtype, array.dtype)

                # We now go to check if original array elements are preserved
                array_size = array.shape[0] * \
                    (array.shape[1] if len(array.shape) > 1 else 1)
                res_size = res.shape[0] * \
                    (res.shape[1] if len(res.shape) > 1 else 1)

                if res_size == 0:
                    self.assertFalse((res != CArray([])).any())
                    return

                array_ravel = CArray(array.ravel())
                if array_ravel.size > res_size:
                    array_ravel = array_ravel[:res_size]
                res_ravel = CArray(res.ravel())
                res_added = None
                if res_ravel.size > array_size:
                    res_added = res_ravel[array_size:]
                    res_ravel = res_ravel[:array_size]

                self.assertFalse((array_ravel != res_ravel).any())
                if res_added is not None:
                    self.assertFalse(
                        (res_added != array.dtype.type(constant)).any())

        _check_resize(self.array_dense, (2, 6))
        _check_resize(self.array_dense, (2, 4))
        _check_resize(self.array_dense, (4, 5))
        _check_resize(self.array_dense, 6)
        _check_resize(self.array_dense, 15)

        _check_resize(self.array_dense_bool, (2, 6))
        _check_resize(self.array_dense_bool, (2, 4))
        _check_resize(self.array_dense_bool, (4, 5))
        _check_resize(self.array_dense_bool, 6)
        _check_resize(self.array_dense_bool, 15)

        _check_resize(self.row_flat_dense, (3, 1))
        _check_resize(self.row_flat_dense, (2, 4))
        _check_resize(self.row_flat_dense, 2)
        _check_resize(self.row_flat_dense, 5)
        _check_resize(self.row_dense, (3, 1))
        _check_resize(self.row_dense, (2, 4))
        _check_resize(self.row_dense, 2)
        _check_resize(self.row_dense, 5)

        _check_resize(self.column_dense, (1, 3))
        _check_resize(self.column_dense, (2, 4))
        _check_resize(self.column_dense, 2)
        _check_resize(self.column_dense, 5)

        _check_resize(self.single_flat_dense, (1, 1))
        _check_resize(self.single_flat_dense, (2, 4))
        _check_resize(self.single_flat_dense, 0)
        _check_resize(self.single_flat_dense, 5)
        _check_resize(self.single_dense, (1, 1))
        _check_resize(self.single_dense, (2, 4))
        _check_resize(self.single_dense, 0)
        _check_resize(self.single_dense, 5)

        _check_resize(self.single_bool_flat_dense, (1, 1))
        _check_resize(self.single_bool_flat_dense, (2, 4))
        _check_resize(self.single_bool_flat_dense, 0)
        _check_resize(self.single_bool_flat_dense, 5)
        _check_resize(self.single_bool_dense,(1, 1))
        _check_resize(self.single_bool_dense, (2, 4))
        _check_resize(self.single_bool_dense, 0)
        _check_resize(self.single_bool_dense, 5)

        _check_resize(self.empty_flat_dense, (1, 1))
        _check_resize(self.empty_flat_dense, (2, 4))
        _check_resize(self.empty_flat_dense, 0)
        _check_resize(self.empty_flat_dense, 5)
        _check_resize(self.empty_dense, (1, 1))
        _check_resize(self.empty_dense, (2, 4))
        _check_resize(self.empty_dense, 0)
        _check_resize(self.empty_dense, 5)

        with self.assertRaises(NotImplementedError):
            self.array_sparse.resize((2, 6))
    

if __name__ == '__main__':
    CArrayTestCases.main()
