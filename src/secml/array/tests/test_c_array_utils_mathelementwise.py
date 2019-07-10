from secml.array.tests import CArrayTestCases

from secml.array import CArray
from secml.core.constants import nan, inf


class TestCArrayUtilsMathElementWise(CArrayTestCases):
    """Unit test for CArray UTILS - MATH ELEMENT-WISE methods."""

    def test_sqrt(self):
        """Test for CArray.sqrt() method."""
        self.logger.info("Test for CArray.sqrt() method.")

        # We are testing sqrt on negative values
        self.logger.filterwarnings(
            action="ignore",
            message="invalid value encountered in sqrt",
            category=RuntimeWarning
        )

        def _check_sqrt(array, expected):
            self.logger.info("Array:\n{:}".format(array))

            res = array.sqrt()
            self.logger.info("array.sqrt():\n{:}".format(res))

            self.assertIsInstance(res, CArray)
            self.assertEqual(res.isdense, expected.isdense)
            self.assertEqual(res.issparse, expected.issparse)
            self.assertEqual(res.shape, expected.shape)
            self.assertEqual(res.dtype, expected.dtype)
            self.assert_array_almost_equal(res, expected, decimal=4)

        # array_dense = CArray([[1, 0, 0, 5], [2, 4, 0, 0], [3, 6, 0, 0]]
        # row_flat_dense = CArray([4, 0, 6])

        _check_sqrt(self.array_dense,
                    CArray([[1., 0., 0., 2.2361],
                            [1.4142, 2., 0., 0.],
                            [1.7320, 2.4495, 0., 0.]]))
        _check_sqrt(self.array_sparse,
                    CArray([[1., 0., 0., 2.2361],
                            [1.4142, 2., 0., 0.],
                            [1.7320, 2.4495, 0., 0.]], tosparse=True))
        _check_sqrt(self.row_flat_dense, CArray([2., 0., 2.4495]))
        _check_sqrt(CArray([4., 0., -3.]), CArray([2., 0., nan]))
        _check_sqrt(self.row_dense, CArray([[2., 0., 2.4495]]))
        _check_sqrt(self.row_sparse, CArray([[2., 0., 2.4495]], tosparse=True))
        _check_sqrt(CArray([[4., 0., -3.]]), CArray([[2., 0., nan]]))
        _check_sqrt(CArray([4., 0., -3.], tosparse=True),
                    CArray([[2., 0., nan]], tosparse=True))
        _check_sqrt(self.column_dense, CArray([[2.], [0.], [2.4495]]))
        _check_sqrt(self.column_sparse,
                    CArray([[2.], [0.], [2.4495]], tosparse=True))
        _check_sqrt(self.single_flat_dense, CArray([2.]))
        _check_sqrt(self.single_dense, CArray([[2.]]))
        _check_sqrt(self.single_sparse, CArray([[2.]], tosparse=True))

    def test_sin(self):
        """Test for CArray.sin() method."""
        self.logger.info("Test for CArray.sin() method.")

        def _check_sin(array, expected):
            self.logger.info("Array:\n{:}".format(array))

            res = array.sin().round(4)
            self.logger.info("array.sin():\n{:}".format(res))

            self.assertIsInstance(res, CArray)
            self.assertEqual(res.isdense, expected.isdense)
            self.assertEqual(res.issparse, expected.issparse)
            self.assertEqual(res.shape, expected.shape)
            self.assertEqual(res.dtype, expected.dtype)
            self.assertFalse((res != expected).any())

        # array_dense = CArray([[1, 0, 0, 5], [2, 4, 0, 0], [3, 6, 0, 0]]
        # row_flat_dense = CArray([4, 0, 6])

        # We consider the values already in radians
        _check_sin(self.array_dense,
                   CArray([[0.8415, 0, 0, -0.9589],
                           [0.9093, -0.7568, 0, 0],
                           [0.1411, -0.2794, 0, 0]]))
        _check_sin(self.row_flat_dense, CArray([-0.7568, 0, -0.2794]))
        _check_sin(self.row_dense, CArray([[-0.7568, 0, -0.2794]]))
        _check_sin(self.column_dense, CArray([[-0.7568], [0], [-0.2794]]))
        _check_sin(self.single_flat_dense, CArray([-0.7568]))
        _check_sin(self.single_dense, CArray([[-0.7568]]))

    def test_cos(self):
        """Test for CArray.cos() method."""
        self.logger.info("Test for CArray.cos() method.")

        def _check_cos(array, expected):
            self.logger.info("Array:\n{:}".format(array))

            res = array.cos().round(4)
            self.logger.info("array.cos():\n{:}".format(res))

            self.assertIsInstance(res, CArray)
            self.assertEqual(res.isdense, expected.isdense)
            self.assertEqual(res.issparse, expected.issparse)
            self.assertEqual(res.shape, expected.shape)
            self.assertEqual(res.dtype, expected.dtype)
            self.assertFalse((res != expected).any())

        # array_dense = CArray([[1, 0, 0, 5], [2, 4, 0, 0], [3, 6, 0, 0]]
        # row_flat_dense = CArray([4, 0, 6])

        # We consider the values already in radians
        _check_cos(self.array_dense,
                   CArray([[0.5403, 1., 1., 0.2837],
                           [-0.4161, -0.6536, 1., 1.],
                           [-0.9900, 0.9602, 1., 1.]]))
        _check_cos(self.row_flat_dense, CArray([-0.6536, 1., 0.9602]))
        _check_cos(self.row_dense, CArray([[-0.6536, 1., 0.9602]]))
        _check_cos(self.column_dense, CArray([[-0.6536], [1.], [0.9602]]))
        _check_cos(self.single_flat_dense, CArray([-0.6536]))
        _check_cos(self.single_dense, CArray([[-0.6536]]))

    def test_exp(self):
        """Test for CArray.exp() method."""
        self.logger.info("Test for CArray.exp() method.")

        def _check_exp(array, expected):
            self.logger.info("Array:\n{:}".format(array))

            res = array.exp().round(4)
            self.logger.info("array.exp():\n{:}".format(res))

            self.assertIsInstance(res, CArray)
            self.assertEqual(res.isdense, expected.isdense)
            self.assertEqual(res.issparse, expected.issparse)
            self.assertEqual(res.shape, expected.shape)
            self.assertEqual(res.dtype, expected.dtype)
            self.assertFalse((res != expected).any())

        # array_dense = CArray([[1, 0, 0, 5], [2, 4, 0, 0], [3, 6, 0, 0]]
        # row_flat_dense = CArray([4, 0, 6])

        _check_exp(self.array_dense,
                   CArray([[2.7183, 1., 1., 148.4132],
                           [7.3891, 54.5982, 1., 1.],
                           [20.0855, 403.4288, 1., 1.]]))
        _check_exp(self.row_flat_dense, CArray([54.5982, 1., 403.4288]))
        _check_exp(self.row_dense, CArray([[54.5982, 1., 403.4288]]))
        _check_exp(self.column_dense, CArray([[54.5982], [1.], [403.4288]]))
        _check_exp(self.single_flat_dense, CArray([54.5982]))
        _check_exp(self.single_dense, CArray([[54.5982]]))

    def test_log(self):
        """Test for CArray.log() method."""
        self.logger.info("Test for CArray.log() method.")

        # We are testing log on zero values
        self.logger.filterwarnings(
            action="ignore",
            message="divide by zero encountered in log",
            category=RuntimeWarning
        )

        def _check_log(array, expected):
            self.logger.info("Array:\n{:}".format(array))

            res = array.log().round(4)
            self.logger.info("array.log():\n{:}".format(res))

            self.assertIsInstance(res, CArray)
            self.assertEqual(res.isdense, expected.isdense)
            self.assertEqual(res.issparse, expected.issparse)
            self.assertEqual(res.shape, expected.shape)
            self.assertEqual(res.dtype, expected.dtype)
            self.assertFalse((res != expected).any())

        # array_dense = CArray([[1, 0, 0, 5], [2, 4, 0, 0], [3, 6, 0, 0]]
        # row_flat_dense = CArray([4, 0, 6])

        _check_log(self.array_dense,
                   CArray([[0., -inf, -inf, 1.6094],
                           [0.6931, 1.3863, -inf, -inf],
                           [1.0986, 1.7918, -inf, -inf]]))
        _check_log(self.row_flat_dense, CArray([1.3863, -inf, 1.7918]))
        _check_log(self.row_dense, CArray([[1.3863, -inf, 1.7918]]))
        _check_log(self.column_dense, CArray([[1.3863], [-inf], [1.7918]]))
        _check_log(self.single_flat_dense, CArray([1.3863]))
        _check_log(self.single_dense, CArray([[1.3863]]))

    def test_log10(self):
        """Test for CArray.log10() method."""
        self.logger.info("Test for CArray.log10() method.")

        # We are testing log10 on zero values
        self.logger.filterwarnings(
            action="ignore",
            message="divide by zero encountered in log10",
            category=RuntimeWarning
        )

        def _check_log10(array, expected):
            self.logger.info("Array:\n{:}".format(array))

            res = array.log10().round(4)
            self.logger.info("array.log10():\n{:}".format(res))

            self.assertIsInstance(res, CArray)
            self.assertEqual(res.isdense, expected.isdense)
            self.assertEqual(res.issparse, expected.issparse)
            self.assertEqual(res.shape, expected.shape)
            self.assertEqual(res.dtype, expected.dtype)
            self.assertFalse((res != expected).any())

        # array_dense = CArray([[1, 0, 0, 5], [2, 4, 0, 0], [3, 6, 0, 0]]
        # row_flat_dense = CArray([4, 0, 6])

        _check_log10(self.array_dense,
                     CArray([[0., -inf, -inf, 0.6990],
                             [0.3010, 0.6021, -inf, -inf],
                             [0.4771, 0.7782, -inf, -inf]]))
        _check_log10(self.row_flat_dense, CArray([0.6021, -inf, 0.7782]))
        _check_log10(self.row_dense, CArray([[0.6021, -inf, 0.7782]]))
        _check_log10(self.column_dense,
                     CArray([[0.6021], [-inf], [0.7782]]))
        _check_log10(self.single_flat_dense, CArray([0.6021]))
        _check_log10(self.single_dense, CArray([[0.6021]]))
    

if __name__ == '__main__':
    CArrayTestCases.main()
