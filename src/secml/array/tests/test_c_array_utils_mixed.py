from secml.array.tests import CArrayTestCases

from secml.array import CArray
from secml.array.c_dense import CDense
from secml.array.c_sparse import CSparse


class TestCArrayUtilsMixed(CArrayTestCases):
    """Unit test for CArray UTILS - MIXED methods."""

    def test_sign(self):
        """Test for CArray.sign() method."""
        self.logger.info("Test for CArray.sign() method.")

        def _check_sign(array, expected):

            for dt in [int, float]:

                array = array.astype(dt)
                expected = expected.astype(dt)

                self.logger.info("Array:\n{:}".format(array))

                res = array.sign()
                self.logger.info("array.sign():\n{:}".format(res))

                self.assertIsInstance(res, CArray)
                self.assertEqual(res.isdense, expected.isdense)
                self.assertEqual(res.issparse, expected.issparse)
                self.assertEqual(res.shape, expected.shape)
                self.assertEqual(res.dtype, expected.dtype)
                self.assertFalse((res != expected).any())

        # array_dense = CArray([[1, 0, 0, 5], [2, 4, 0, 0], [3, 6, 0, 0]]
        # row_flat_dense = CArray([4, 0, 6])

        # DENSE
        data = self.array_dense
        data[2, :] *= -1
        _check_sign(data, CArray([[1, 0, 0, 1], [1, 1, 0, 0], [-1, -1, 0, 0]]))
        _check_sign(CArray([4, 0, -6]), CArray([1, 0, -1]))
        _check_sign(CArray([[4, 0, -6]]), CArray([[1, 0, -1]]))
        _check_sign(CArray([[4, 0, -6]]).T, CArray([[1], [0], [-1]]))
        _check_sign(CArray([4]), CArray([1]))
        _check_sign(CArray([0]), CArray([0]))
        _check_sign(CArray([-4]), CArray([-1]))
        _check_sign(CArray([[4]]), CArray([[1]]))
        _check_sign(CArray([[0]]), CArray([[0]]))
        _check_sign(CArray([[-4]]), CArray([[-1]]))

        # SPARSE
        data = self.array_sparse
        data[2, :] *= -1
        _check_sign(data, CArray([[1, 0, 0, 1], [1, 1, 0, 0], [-1, -1, 0, 0]],
                                 tosparse=True))
        _check_sign(CArray([[4, 0, -6]], tosparse=True),
                    CArray([[1, 0, -1]], tosparse=True))
        _check_sign(CArray([[4, 0, -6]], tosparse=True).T,
                    CArray([[1], [0], [-1]], tosparse=True))
        _check_sign(CArray([4], tosparse=True), CArray([1], tosparse=True))
        _check_sign(CArray([0], tosparse=True), CArray([0], tosparse=True))
        _check_sign(CArray([-4], tosparse=True), CArray([-1], tosparse=True))

        # BOOL
        _check_sign(self.array_dense_bool,
                    CArray([[1, 0, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]]))
        _check_sign(self.array_sparse_bool,
                    CArray([[1, 0, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]],
                           tosparse=True))
        _check_sign(CArray([True]), CArray([1]))
        _check_sign(CArray([False]), CArray([0]))
        _check_sign(CArray([[True]]), CArray([[1]]))
        _check_sign(CArray([[False]]), CArray([[0]]))
        _check_sign(CArray([[True]], tosparse=True),
                    CArray([[1]], tosparse=True))
        _check_sign(CArray([[False]], tosparse=True),
                    CArray([[0]], tosparse=True))

    def test_diag(self):
        """Test for CArray.diag() method."""
        self.logger.info("Test for CArray.diag() method.")

        def extract_diag(array, k, out):

            diag = array.diag(k=k)
            self.logger.info("({:})-th diagonal is: {:}".format(k, diag))
            self.assertEqual(1, diag.ndim)
            self.assertTrue(diag.isdense)
            self.assertTrue((diag == out).all())

        self.logger.info("Testing diagonal extraction...")

        self.logger.info("Array is:\n{:}".format(self.array_dense))

        extract_diag(self.array_dense, k=0, out=CArray([1, 4, 0]))
        extract_diag(self.array_dense, k=1, out=CArray([0, 0, 0]))
        extract_diag(self.array_dense, k=-1, out=CArray([2, 6]))

        with self.assertRaises(ValueError):
            # k is higher/lower than array shape
            self.array_dense.diag(k=4)
        with self.assertRaises(ValueError):
            # k is higher/lower than array shape
            self.array_dense.diag(k=-3)

        self.logger.info("Array is:\n{:}".format(self.array_sparse))

        extract_diag(self.array_sparse, k=0, out=CArray([1, 4, 0]))
        extract_diag(self.array_sparse, k=1, out=CArray([0, 0, 0]))
        extract_diag(self.array_sparse, k=-1, out=CArray([2, 6]))

        with self.assertRaises(ValueError):
            # k is higher/lower than array shape
            self.array_sparse.diag(k=4)
        with self.assertRaises(ValueError):
            # k is higher/lower than array shape
            self.array_sparse.diag(k=-3)

        self.logger.info("Testing diagonal array creation...")

        def create_diag(array, k, out):

            diag = array.diag(k=k)
            self.logger.info(
                "Array created using k={:} is:\n{:}".format(k, diag))
            self.assertEqual(array.isdense, diag.isdense)
            self.assertEqual(array.issparse, diag.issparse)
            self.assertTrue((diag == out).all())

        self.logger.info("Array is:\n{:}".format(self.row_flat_dense))

        out_diag = CArray([[4, 0, 0], [0, 0, 0], [0, 0, 6]])
        create_diag(self.row_flat_dense, k=0, out=out_diag)

        out_diag = CArray([[0, 4, 0, 0], [0, 0, 0, 0],
                           [0, 0, 0, 6], [0, 0, 0, 0]])
        create_diag(self.row_flat_dense, k=1, out=out_diag)

        self.logger.info("Array is:\n{:}".format(self.row_dense))

        out_diag = CArray([[4, 0, 0], [0, 0, 0], [0, 0, 6]])
        create_diag(self.row_dense, k=0, out=out_diag)

        out_diag = CArray([[0, 4, 0, 0], [0, 0, 0, 0],
                           [0, 0, 0, 6], [0, 0, 0, 0]])
        create_diag(self.row_dense, k=1, out=out_diag)

        self.logger.info("Array is:\n{:}".format(self.row_sparse))

        out_diag = CArray([[4, 0, 0], [0, 0, 0], [0, 0, 6]])
        create_diag(self.row_sparse, k=0, out=out_diag)

        out_diag = CArray([[0, 4, 0, 0], [0, 0, 0, 0],
                           [0, 0, 0, 6], [0, 0, 0, 0]])
        create_diag(self.row_sparse, k=1, out=out_diag)

        self.logger.info("Testing diagonal array creation from single val...")

        self.logger.info("Array is:\n{:}".format(self.single_flat_dense))

        create_diag(self.single_flat_dense, k=0, out=CArray([[4]]))
        create_diag(self.single_flat_dense, k=1, out=CArray([[0, 4], [0, 0]]))

        self.logger.info("Array is:\n{:}".format(self.single_dense))

        create_diag(self.single_dense, k=0, out=CArray([[4]]))
        create_diag(self.single_dense, k=1, out=CArray([[0, 4], [0, 0]]))

        self.logger.info("Array is:\n{:}".format(self.single_sparse))

        create_diag(self.single_sparse, k=0, out=CArray([[4]]))
        create_diag(self.single_sparse, k=1, out=CArray([[0, 4], [0, 0]]))

        self.logger.info("Testing diagonal returns error on empty arrays...")

        with self.assertRaises(ValueError):
            self.empty_flat_dense.diag()

        with self.assertRaises(ValueError):
            self.empty_sparse.diag()

    def test_dot(self):
        """"Test for CArray.dot() method."""
        self.logger.info("Test for CArray.dot() method.")
        s_vs_s = self.array_sparse.dot(self.array_sparse.T)
        s_vs_d = self.array_sparse.dot(self.array_dense.T)
        d_vs_d = self.array_dense.dot(self.array_dense.T)
        d_vs_s = self.array_dense.dot(self.array_sparse.T)

        # Check if method returned correct datatypes
        self.assertIsInstance(s_vs_s._data, CSparse)
        self.assertIsInstance(s_vs_d._data, CSparse)
        self.assertIsInstance(d_vs_d._data, CDense)
        self.assertIsInstance(d_vs_s._data, CDense)

        # Check if we have the same output in all cases
        self.assertTrue(
            self._test_multiple_eq([s_vs_s, s_vs_d, d_vs_d, d_vs_s]))

        # Test inner product between vector-like arrays
        def _check_dot_vector_like(array1, array2, expected):
            dot_res = array1.dot(array2)
            self.logger.info("We made a dot between {:} and {:}, "
                             "result: {:}.".format(array1, array2, dot_res))
            self.assertEqual(dot_res, expected)

        _check_dot_vector_like(self.row_flat_dense, self.column_dense, 52)
        _check_dot_vector_like(self.row_flat_dense, self.column_sparse, 52)
        _check_dot_vector_like(self.row_dense, self.column_dense, 52)
        _check_dot_vector_like(self.row_dense, self.column_sparse, 52)
        _check_dot_vector_like(self.row_sparse, self.column_dense, 52)
        _check_dot_vector_like(self.row_sparse, self.column_sparse, 52)

        dense_flat_outer = self.column_dense.dot(self.row_flat_dense)
        self.logger.info("We made a dot between {:} and {:}, "
                         "result: {:}.".format(self.column_dense,
                                               self.row_flat_dense,
                                               dense_flat_outer))
        self.assertEqual(len(dense_flat_outer.shape), 2,
                         "Dot result column.dot(row) is not a matrix!")

        # Test between flats
        dot_res_flats = CArray([10, 20]).dot(CArray([1, 0]))
        self.assertEqual(dot_res_flats, 10)


if __name__ == '__main__':
    CArrayTestCases.main()
