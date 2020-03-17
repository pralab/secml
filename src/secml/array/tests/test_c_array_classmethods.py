from secml.array.tests import CArrayTestCases

from secml.array import CArray


class TestCArrayClassMethods(CArrayTestCases):
    """Unit test for CArray CLASSMETHODS methods."""

    def test_concatenate(self):
        """Test for CArray.concatenate() method."""
        self.logger.info("Test for CArray.concatenate() method.")

        def _concat_allaxis(array1, array2):

            self.logger.info("a1: {:} ".format(array1))
            self.logger.info("a2: {:} ".format(array2))

            # check concatenate, axis None (ravelled)
            concat_res = CArray.concatenate(array1, array2, axis=None)
            self.logger.info("concat(a1, a2): {:}".format(concat_res))
            # If axis is None, result should be ravelled...
            if array1.isdense:
                self.assertEqual(1, concat_res.ndim)
            else:  # ... but if array is sparse let's check for shape[0]
                self.assertEqual(1, concat_res.shape[0])
            # Let's check the elements of the resulting array
            a1_comp = array1.todense().ravel()
            a2_comp = array2.todense().ravel()
            if array1.issparse:  # result will be sparse, so always 2d
                a1_comp = a1_comp.atleast_2d()
                a2_comp = a2_comp.atleast_2d()
            self.assert_array_equal(concat_res[:array1.size], a1_comp)
            self.assert_array_equal(concat_res[array1.size:], a2_comp)

            array1_shape0 = array1.atleast_2d().shape[0]
            array1_shape1 = array1.atleast_2d().shape[1]
            array2_shape0 = array2.atleast_2d().shape[0]
            array2_shape1 = array2.atleast_2d().shape[1]

            # check append on axis 0 (vertical)
            concat_res = CArray.concatenate(array1, array2, axis=0)
            self.logger.info("concat(a1, a2, axis=0): {:}".format(concat_res))
            self.assertEqual(array1_shape1, concat_res.shape[1])
            self.assertEqual(
                array1_shape0 + array2_shape0, concat_res.shape[0])
            self.assert_array_equal(concat_res[:array1_shape0, :], array1)
            self.assert_array_equal(concat_res[array1_shape0:, :], array2)

            # check append on axis 1 (horizontal)
            concat_res = CArray.concatenate(array1, array2, axis=1)
            self.logger.info("concat(a1, a2, axis=1): {:}".format(concat_res))
            self.assertEqual(
                array1_shape1 + array2_shape1, concat_res.shape[1])
            self.assertEqual(array1_shape0, concat_res.shape[0])
            self.assert_array_equal(concat_res[:, :array1_shape1], array1)
            self.assert_array_equal(concat_res[:, array1_shape1:], array2)

        _concat_allaxis(self.array_dense, self.array_dense)
        _concat_allaxis(self.array_sparse, self.array_sparse)
        _concat_allaxis(self.array_sparse, self.array_dense)
        _concat_allaxis(self.array_dense, self.array_sparse)

        # check concat on empty arrays
        empty_sparse = CArray([], tosparse=True)
        empty_dense = CArray([], tosparse=False)
        self.assertTrue((CArray.concatenate(
                empty_sparse, empty_dense, axis=None) == empty_dense).all())
        self.assertTrue((CArray.concatenate(
                empty_sparse, empty_dense, axis=0) == empty_dense).all())
        self.assertTrue((CArray.concatenate(
                empty_sparse, empty_dense, axis=1) == empty_dense).all())

    def test_comblist(self):
        """Test for comblist() classmethod."""
        self.logger.info("Test for comblist() classmethod.")

        l = [[1, 2], [4]]
        self.logger.info("list of lists: \n{:}".format(l))
        comb_array = CArray.comblist(l)
        self.logger.info("comblist(l): \n{:}".format(comb_array))
        self.assertTrue((comb_array == CArray([[1., 4.], [2., 4.]])).all())

        l = [[1, 2], []]
        self.logger.info("list of lists: \n{:}".format(l))
        comb_array = CArray.comblist(l)
        self.logger.info("comblist(l): \n{:}".format(comb_array))
        self.assertTrue((comb_array == CArray([[1.], [2.]])).all())

        l = [[], []]
        comb_array = CArray.comblist(l)
        self.logger.info("comblist(l): \n{:}".format(comb_array))
        self.assertTrue((comb_array == CArray([])).all())

    def test_fromiterables(self):
        """Test for CArray.from_iterables classmethod."""
        self.logger.info("Test for CArray.from_iterables classmethod.")

        expected = CArray([1, 2, 3, 4, 5, 6])

        a = CArray.from_iterables([[1, 2], (3, 4), CArray([5, 6])])
        self.logger.info("from_iterables result: {:}".format(a))
        self.assertFalse((a != expected).any())

        a = CArray.from_iterables([CArray([1, 2]), CArray([[3, 4], [5, 6]])])
        self.logger.info("from_iterables result: {:}".format(a))
        self.assertFalse((a != expected).any())

        a = CArray.from_iterables([CArray([1, 2, 3, 4, 5, 6]), CArray([])])
        self.logger.info("from_iterables result: {:}".format(a))
        self.assertFalse((a != expected).any())

        a = CArray.from_iterables([CArray([1, 2, 3, 4, 5, 6]), []])
        self.logger.info("from_iterables result: {:}".format(a))
        self.assertFalse((a != expected).any())

        a = CArray.from_iterables([(), CArray([1, 2, 3, 4, 5, 6])])
        self.logger.info("from_iterables result: {:}".format(a))
        self.assertFalse((a != expected).any())

        a = CArray.from_iterables([CArray([[1, 2, 3, 4, 5, 6]])])
        self.logger.info("from_iterables result: {:}".format(a))
        self.assertFalse((a != expected).any())

        a = CArray.from_iterables(
            [CArray([[1, 2, 3, 4, 5, 6]], tosparse=True)])
        self.logger.info("from_iterables result: {:}".format(a))
        self.assertFalse((a != expected).any())

    def test_ones(self):
        """Test for CArray.ones() classmethod."""
        self.logger.info("Test for CArray.ones() classmethod.")

        for shape in [1, (1, ), 2, (2, ), (1, 2), (2, 1), (2, 2)]:
            for dtype in [None, float, int, bool]:
                for sparse in [False, True]:
                    res = CArray.ones(shape=shape, dtype=dtype, sparse=sparse)
                    self.logger.info(
                        "CArray.ones(shape={:}, dtype={:}, sparse={:}):"
                        "\n{:}".format(shape, dtype, sparse, res))

                    self.assertIsInstance(res, CArray)
                    self.assertEqual(res.isdense, not sparse)
                    self.assertEqual(res.issparse, sparse)
                    if isinstance(shape, tuple):
                        if len(shape) == 1 and sparse is True:
                            # Sparse "vectors" have len(shape) == 2
                            self.assertEqual(res.shape, (1, shape[0]))
                        else:
                            self.assertEqual(res.shape, shape)
                    else:
                        if sparse is True:
                            self.assertEqual(res.shape, (1, shape))
                        else:
                            self.assertEqual(res.shape, (shape, ))
                    if dtype is None:  # Default dtype is float
                        self.assertEqual(res.dtype, float)
                    else:
                        self.assertEqual(res.dtype, dtype)
                    self.assertTrue((res == 1).all())

    def test_zeros(self):
        """Test for CArray.zeros() classmethod."""
        self.logger.info("Test for CArray.zeros() classmethod.")

        for shape in [1, (1, ), 2, (2, ), (1, 2), (2, 1), (2, 2)]:
            for dtype in [None, float, int, bool]:
                for sparse in [False, True]:
                    res = CArray.zeros(shape=shape, dtype=dtype, sparse=sparse)
                    self.logger.info(
                        "CArray.zeros(shape={:}, dtype={:}, sparse={:}):"
                        "\n{:}".format(shape, dtype, sparse, res))

                    self.assertIsInstance(res, CArray)
                    self.assertEqual(res.isdense, not sparse)
                    self.assertEqual(res.issparse, sparse)
                    if isinstance(shape, tuple):
                        if len(shape) == 1 and sparse is True:
                            # Sparse "vectors" have len(shape) == 2
                            self.assertEqual(res.shape, (1, shape[0]))
                        else:
                            self.assertEqual(res.shape, shape)
                    else:
                        if sparse is True:
                            self.assertEqual(res.shape, (1, shape))
                        else:
                            self.assertEqual(res.shape, (shape, ))
                    if dtype is None:  # Default dtype is float
                        self.assertEqual(res.dtype, float)
                    else:
                        self.assertEqual(res.dtype, dtype)
                    self.assertFalse((res != 0).any())

    def test_empty(self):
        """Test for CArray.empty() classmethod."""
        self.logger.info("Test for CArray.empty() classmethod.")

        for shape in [1, (1, ), 2, (2, ), (1, 2), (2, 1), (2, 2)]:
            for dtype in [None, float, int, bool]:
                for sparse in [False, True]:
                    res = CArray.empty(shape=shape, dtype=dtype, sparse=sparse)
                    self.logger.info(
                        "CArray.empty(shape={:}, dtype={:}, sparse={:}):"
                        "\n{:}".format(shape, dtype, sparse, res))

                    self.assertIsInstance(res, CArray)
                    self.assertEqual(res.isdense, not sparse)
                    self.assertEqual(res.issparse, sparse)
                    if isinstance(shape, tuple):
                        if len(shape) == 1 and sparse is True:
                            # Sparse "vectors" have len(shape) == 2
                            self.assertEqual(res.shape, (1, shape[0]))
                        else:
                            self.assertEqual(res.shape, shape)
                    else:
                        if sparse is True:
                            self.assertEqual(res.shape, (1, shape))
                        else:
                            self.assertEqual(res.shape, (shape, ))
                    if dtype is None:  # Default dtype is float
                        self.assertEqual(res.dtype, float)
                    else:
                        self.assertEqual(res.dtype, dtype)

                    # NOTE: values of empty arrays are random so
                    # we cannot check the result content

    def test_eye(self):
        """Test for CArray.eye() classmethod."""
        self.logger.info("Test for CArray.eye() classmethod.")

        for dtype in [None, float, int, bool]:
            for sparse in [False, True]:
                for n_rows in [0, 1, 2, 3]:
                    for n_cols in [None, 0, 1, 2, 3]:
                        for k in [0, 1, 2, 3, -1, -2, -3]:
                            res = CArray.eye(n_rows=n_rows, n_cols=n_cols, k=k,
                                             dtype=dtype, sparse=sparse)
                            self.logger.info(
                                "CArray.eye(n_rows={:}, n_cols={:}, k={:}, "
                                "dtype={:}, sparse={:}):\n{:}".format(
                                    n_rows, n_cols, k, dtype, sparse, res))

                            self.assertIsInstance(res, CArray)
                            self.assertEqual(res.isdense, not sparse)
                            self.assertEqual(res.issparse, sparse)

                            if dtype is None:  # Default dtype is float
                                self.assertEqual(res.dtype, float)
                            else:
                                self.assertEqual(res.dtype, dtype)

                            # n_cols takes n_rows if None
                            n_cols = n_rows if n_cols is None else n_cols
                            self.assertEqual(res.shape, (n_rows, n_cols))

                            # Resulting array has no elements, skip more checks
                            if res.size == 0:
                                continue

                            # Check if the diagonal is moving according to k
                            if k > 0:
                                self.assertEqual(
                                    0, res[0, min(n_cols-1, k-1)].item())
                            elif k < 0:
                                self.assertEqual(
                                    0, res[min(n_rows-1, abs(k)-1), 0].item())
                            else:  # The top left corner is a one
                                self.assertEqual(1, res[0, 0])

                            # Check the number of ones
                            n_ones = (res == 1).sum()
                            if k >= 0:
                                self.assertEqual(
                                    max(0, min(n_rows, n_cols-k)), n_ones)
                            else:
                                self.assertEqual(
                                    max(0, min(n_cols, n_rows-abs(k))), n_ones)

                            # Check if there are other elements apart from 0,1
                            self.assertFalse(
                                ((res != 0).logical_and((res == 1).logical_not()).any()))

    def test_rand(self):
        """Test for CArray.rand() classmethod."""
        self.logger.info("Test for CArray.rand() classmethod.")

        for shape in [(1, ), (2, ), (1, 2), (2, 1), (2, 2)]:
            for sparse in [False, True]:
                res = CArray.rand(shape=shape, sparse=sparse)
                self.logger.info(
                    "CArray.rand(shape={:}, sparse={:}):"
                    "\n{:}".format(shape, sparse, res))

                self.assertIsInstance(res, CArray)
                self.assertEqual(res.isdense, not sparse)
                self.assertEqual(res.issparse, sparse)
                if len(shape) == 1 and sparse is True:
                    # Sparse "vectors" have len(shape) == 2
                    self.assertEqual(res.shape, (1, shape[0]))
                else:
                    self.assertEqual(res.shape, shape)
                self.assertEqual(res.dtype, float)

                # Interval of random values is [0.0, 1.0)
                self.assertFalse((res < 0).any())
                self.assertFalse((res >= 1).any())

    def test_randn(self):
        """Test for CArray.randn() classmethod."""
        self.logger.info("Test for CArray.randn() classmethod.")

        for shape in [(1, ), (2, ), (1, 2), (2, 1), (2, 2)]:
            res = CArray.randn(shape=shape)
            self.logger.info(
                "CArray.randn(shape={:}):\n{:}".format(shape, res))

            self.assertIsInstance(res, CArray)
            self.assertEqual(res.shape, shape)
            self.assertEqual(res.dtype, float)

            # NOTE: values are from a random normal distribution so
            # we cannot check the result content

    def test_randint(self):
        """Test for CArray.randint() classmethod."""
        self.logger.info("Test for CArray.randint() classmethod.")

        for inter in [1, 2, (0, 1), (0, 2), (1, 3)]:
            for shape in [1, 2, (1, 2), (2, 1), (2, 2)]:
                for sparse in [False, True]:
                    if not isinstance(inter, tuple):
                        res = CArray.randint(
                            inter, shape=shape, sparse=sparse)
                    else:
                        res = CArray.randint(
                            *inter, shape=shape, sparse=sparse)
                    self.logger.info(
                        "CArray.randint({:}, shape={:}, sparse={:}):"
                        "\n{:}".format(inter, shape, sparse, res))

                    self.assertIsInstance(res, CArray)
                    self.assertEqual(res.isdense, not sparse)
                    self.assertEqual(res.issparse, sparse)
                    if isinstance(shape, tuple):
                        self.assertEqual(res.shape, shape)
                    else:
                        if sparse is True:
                            self.assertEqual(res.shape, (1, shape))
                        else:
                            self.assertEqual(res.shape, (shape, ))
                    self.assertEqual(res.dtype, int)

                # Checking intervals
                if not isinstance(inter, tuple):
                    self.assertFalse((res < 0).any())
                    self.assertFalse((res >= inter).any())
                else:
                    if inter[0] > 0:
                        # Comparing a sparse matrix with a scalar greater
                        # than zero using < is inefficient, use >=
                        self.assertTrue((res >= inter[0]).all())
                    else:
                        self.assertFalse((res < inter[0]).any())
                    self.assertFalse((res >= inter[1]).any())

    def test_randuniform(self):
        """Test for CArray.randuniform() classmethod."""
        self.logger.info("Test for CArray.randuniform() classmethod.")

        for inter in [(0, 1), (0, 2), (1, 3)]:
            for shape in [1, 2, (1, 2), (2, 1), (2, 2)]:
                for sparse in [False, True]:
                    if not isinstance(inter, tuple):
                        res = CArray.randuniform(
                            inter, shape=shape, sparse=sparse)
                    else:
                        res = CArray.randuniform(
                            *inter, shape=shape, sparse=sparse)
                    self.logger.info(
                        "CArray.randuniform({:}, shape={:}, sparse={:}):"
                        "\n{:}".format(inter, shape, sparse, res))

                    self.assertIsInstance(res, CArray)
                    self.assertEqual(res.isdense, not sparse)
                    self.assertEqual(res.issparse, sparse)
                    if isinstance(shape, tuple):
                        self.assertEqual(res.shape, shape)
                    else:
                        if sparse is True:
                            self.assertEqual(res.shape, (1, shape))
                        else:
                            self.assertEqual(res.shape, (shape, ))
                    self.assertEqual(res.dtype, float)

                # Checking intervals
                if not isinstance(inter, tuple):
                    self.assertFalse((res < 0).any())
                    self.assertFalse((res >= inter).any())
                else:
                    if inter[0] > 0:
                        # Comparing a sparse matrix with a scalar greater
                        # than zero using < is inefficient, use >=
                        self.assertTrue((res >= inter[0]).all())
                    else:
                        self.assertFalse((res < inter[0]).any())
                    self.assertFalse((res >= inter[1]).any())

        # Testing arrays as high/low
        bounds = (CArray([-1, -2, 3]), 5)
        res = CArray.randuniform(*bounds, shape=(2, 3))
        self.assertFalse((res < bounds[0]).any())
        self.assertFalse((res >= bounds[1]).any())

        bounds = (-4, CArray([-1, -2, 3]))
        res = CArray.randuniform(*bounds, shape=(2, 3))
        self.assertFalse((res < bounds[0]).any())
        self.assertFalse((res >= bounds[1]).any())

        bounds = (CArray([-5, -8, 1]), CArray([-1, -2, 3]))
        res = CArray.randuniform(*bounds, shape=(2, 3))
        self.assertFalse((res < bounds[0]).any())
        self.assertFalse((res >= bounds[1]).any())

        # if low is higher then high -> ValueError
        with self.assertRaises(ValueError):
            CArray.randuniform(5, CArray([-1, -2, 3]), (2, 3))
        with self.assertRaises(ValueError):
            CArray.randuniform(CArray([5, -3, 4]), CArray([-1, -2, 3]), (2, 3))


if __name__ == '__main__':
    CArrayTestCases.main()
