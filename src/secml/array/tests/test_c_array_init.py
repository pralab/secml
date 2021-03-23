from secml.array.tests import CArrayTestCases

import numpy as np
import scipy.sparse as scs

from secml.array import CArray
from secml.array.c_dense import CDense
from secml.array.c_sparse import CSparse
from secml.core.type_utils import \
    is_scalar, is_bool, is_list, is_list_of_lists


class TestCArrayInit(CArrayTestCases):
    """Unit test for CArray INIT."""

    def test_init_builtin(self):
        """Test CArray initialization using builtin types."""

        def check_init_builtin(totest_elem):

            for tosparse in [False, True]:
                init_array = CArray(totest_elem, tosparse=tosparse)
                self.assertEqual(init_array.issparse, tosparse)

                if is_list_of_lists(totest_elem):
                    if not is_list_of_lists(totest_elem[0]):
                        self.assertEqual(
                            init_array.shape[0], len(totest_elem))
                        self.assertEqual(
                            init_array.shape[1], len(totest_elem[0]))
                    else:  # N-Dimensional input
                        in_shape = init_array.input_shape
                        self.assertEqual(in_shape[0], len(totest_elem))
                        self.assertEqual(in_shape[1], len(totest_elem[0]))
                        self.assertEqual(
                            init_array.shape[0], len(totest_elem))
                        self.assertEqual(
                            init_array.shape[1], sum(in_shape[1:]))

                elif is_list(totest_elem):
                    if init_array.issparse is True:
                        self.assertEqual(
                            init_array.shape[1], len(totest_elem))
                    elif init_array.isdense is True:
                        self.assertTrue(init_array.ndim == 1)
                        self.assertEqual(
                            init_array.shape[0], len(totest_elem))
                    self.assertEqual(
                        init_array.input_shape, (len(totest_elem), ))

                elif is_scalar(totest_elem) or is_bool(totest_elem):
                    self.assertEqual(init_array.size, 1)
                    self.assertEqual(init_array.input_shape, (1, ))

                else:
                    raise TypeError(
                        "test_init_builtin should not be used "
                        "to test {:}".format(type(totest_elem)))

        self.logger.info("Initializing CArray with built-in types...")
        check_init_builtin([[2, 3], [22, 33]])
        check_init_builtin([2, 3])
        check_init_builtin([[2], [3]])
        check_init_builtin(3)
        check_init_builtin(True)
        check_init_builtin([[True, False], [True, True]])
        check_init_builtin([True, False])
        check_init_builtin([[[2, 3], [22, 33]], [[4, 5], [44, 55]]])
        check_init_builtin([[[True, False], [True, True]],
                            [[False, False], [False, True]]])

        # The following input data is malformed and should raise TypeError
        with self.logger.catch_warnings():
            self.logger.filterwarnings(
                action='ignore',
                message="Creating an ndarray from ragged",
                category=np.VisibleDeprecationWarning)
            with self.assertRaises(TypeError):
                CArray([[2, 3], [22]])
            with self.assertRaises(TypeError):
                CArray([[[2, 3], [22]], [[4, 5], [44, 55]]])

    def test_init_array(self):
        """Test CArray initialization using arrays."""

        self.logger.info("Initializing CArray with another CArray...")
        arrays_list = [CArray([[2, 3], [22, 33]]),
                       CArray([2, 3]),
                       CArray([[2], [3]]),
                       CArray(3),
                       CArray([[[2, 3], [22, 33]], [[4, 5], [44, 55]]])]

        for init_elem in arrays_list:
            self.logger.info(init_elem)
            array = CArray(init_elem)
            self.assertEqual(init_elem.issparse, array.issparse)
            self.assert_array_equal(array, init_elem)
            self.assertEqual(init_elem.shape, array.shape)
            self.assertEqual(init_elem.input_shape, array.input_shape)

        self.logger.info("Initializing CArray with a CDense...")
        dense_list = [CDense([[2, 3], [22, 33]]),
                      CDense([2, 3]),
                      CDense([[2], [3]]),
                      CDense([3]),
                      CDense([[[2, 3], [22, 33]], [[4, 5], [44, 55]]])]

        for init_elem in dense_list:
            self.logger.info(init_elem)
            array = CArray(init_elem)
            self.assertTrue(array.isdense)
            self.assert_array_equal(array, init_elem)
            self.assertEqual(array.shape, init_elem.shape)
            self.assertEqual(init_elem.input_shape, array.input_shape)

        self.logger.info("Initializing CArray with an ndarray...")
        dense_list = [np.array([[2, 3], [22, 33]]),
                      np.array([2, 3]),
                      np.array([[2], [3]]),
                      np.array([3]),
                      np.array([[[2, 3], [22, 33]], [[4, 5], [44, 55]]])]

        for init_elem in dense_list:
            self.logger.info(init_elem)
            array = CArray(init_elem)
            self.assertTrue(array.isdense)
            if init_elem.ndim <= 2:
                self.assert_array_equal(array, init_elem)
                self.assertEqual(array.shape, init_elem.shape)
            else:  # N-Dimensional ndarray
                self.assertEqual(array.shape[0], init_elem.shape[0])
                self.assertEqual(array.shape[1], sum(init_elem.shape[1:]))
            self.assertEqual(array.input_shape, init_elem.shape)

        self.logger.info("Initializing CArray with a sparse CArray...")
        sparse_list = [
            CArray([[2, 3], [22, 33]], tosparse=True),
            CArray([2, 3], tosparse=True),
            CArray([[2], [3]], tosparse=True),
            CArray(3, tosparse=True),
            CArray([[[2, 3], [22, 33]], [[4, 5], [44, 55]]], tosparse=True)]

        for init_elem in sparse_list:
            self.logger.info(init_elem)
            array = CArray(init_elem)
            self.assertTrue(array.issparse)
            self.assert_array_equal(array, init_elem)
            self.assertEqual(init_elem.input_shape, array.input_shape)

        self.logger.info("Initializing CArray with a CSparse...")
        sparse_list = [CSparse([[2, 3], [22, 33]]),
                       CSparse([2, 3]),
                       CSparse([[2], [3]]),
                       CSparse([3]),
                       CSparse([[[2, 3], [22, 33]], [[4, 5], [44, 55]]])]

        for init_elem in sparse_list:
            self.logger.info(init_elem)
            array = CArray(init_elem)
            self.assertTrue(array.issparse)
            self.assert_array_equal(array, init_elem)
            self.assertEqual(array.shape, init_elem.shape)
            self.assertEqual(array.input_shape, init_elem.input_shape)

        self.logger.info("Initializing CArray with a csr_matrix...")
        sparse_list = [scs.csr_matrix([[2, 3], [22, 33]]),
                       scs.csr_matrix([2, 3]),
                       scs.csr_matrix([[2], [3]]),
                       scs.csr_matrix([3])]

        for init_elem in sparse_list:
            self.logger.info(init_elem)
            array = CArray(init_elem)
            self.assertTrue(array.issparse)
            self.assert_array_equal(array.todense(), init_elem.todense())
            self.assertEqual(array.shape, init_elem.shape)
            self.assertEqual(array.input_shape, init_elem.shape)

    def test_init_reshape(self):
        """Test CArray reshape during initialization."""
        arrays = [[[2, 3], [22, 33]], [2, 3], [[2], [3]], 3]

        for a in arrays:
            for sparse in (False, True):
                out_def = CArray(a)
                size = out_def.size  # Expected size
                in_shape = out_def.shape  # Expected input_shape

                for shape in [size, (size, ), (1, size), (size, 1)]:
                    out_res = CArray(a, tosparse=sparse, shape=shape)

                    # Resulting shape will always be (1, n) for sparse
                    if is_scalar(shape):
                        shape = (1, shape) if out_res.issparse else (shape, )
                    if out_res.issparse and len(shape) < 2:
                        shape = (1, shape[0])

                    self.logger.info("Expected 'shape' {:}, got {:}".format(
                        shape, out_res.shape))
                    self.assertEqual(out_res.shape, shape)

                    # The input_shape should not be altered by reshaping
                    self.logger.info(
                        "Expected 'input_shape' {:}, got {:}".format(
                            in_shape, out_res.input_shape))
                    self.assertEqual(out_res.input_shape, in_shape)

                with self.assertRaises(ValueError):
                    # Shape with wrong size, expect error
                    CArray(a, tosparse=sparse, shape=(2, size))

    def test_init_empty(self):
        """Test CArray initialization using empty structures."""
        # Initialization using empty arrays
        empty_init = []
        for test_case in (False, ):
            self.logger.info(
                "Testing flat empty, tosparse: {:}".format(test_case))
            array_empty = CArray(empty_init, tosparse=test_case)
            self.assertEqual(array_empty.size, 0)
            self.assertEqual(array_empty.shape, (0, ))
            self.assertEqual(array_empty.ndim, 1)

        empty_init = []  # Empty sparse arrays are always 2D
        for test_case in (True, ):
            self.logger.info(
                "Testing flat empty, tosparse: {:}".format(test_case))
            array_empty = CArray(empty_init, tosparse=test_case)
            self.assertEqual(array_empty.size, 0)
            self.assertEqual(array_empty.shape, (1, 0))
            self.assertEqual(array_empty.ndim, 2)

        empty_init = [[]]
        for test_case in (False, True):
            self.logger.info(
                "Testing 2D empty, tosparse: {:}".format(test_case))
            array_empty = CArray(empty_init, tosparse=test_case)
            self.assertEqual(array_empty.size, 0)
            self.assertEqual(array_empty.shape, (1, 0))
            self.assertEqual(array_empty.ndim, 2)

    def test_input_shape(self):
        """Test CArray.input_shape behavior."""
        array = CArray([[[2, 3], [22, 33]], [[4, 5], [44, 55]]])
        array_s = \
            CArray([[[2, 3], [22, 33]], [[4, 5], [44, 55]]], tosparse=True)
        ref_shape = (2, 2, 2)

        # not propagate on getitem (as it returns new objects)
        out = array[0:2, 0:2]
        self.assertEqual(out.input_shape, out.shape)
        out = array_s[0:2, 0:2]
        self.assertEqual(out.input_shape, out.shape)

        # not propagate on other generic methods (as they return new objects)
        out = array.astype(float)
        self.assertEqual(out.input_shape, out.shape)
        out = array.unique()
        self.assertEqual(out.input_shape, out.shape)
        out = array.all(axis=0)
        self.assertEqual(out.input_shape, out.shape)

        # not propagate on classmethods (es. concatenate/append)
        out = CArray.concatenate(array, array, axis=0)
        self.assertEqual(out.input_shape, out.shape)
        out = CArray.concatenate(array, array, axis=None)
        self.assertEqual(out.input_shape, out.shape)

        # should propagate on copy/deepcopy
        from copy import copy, deepcopy

        array_c = copy(array)
        self.assertEqual(array_c.input_shape, ref_shape)
        array_c = copy(array_s)
        self.assertEqual(array_c.input_shape, ref_shape)

        array_c = deepcopy(array)
        self.assertEqual(array_c.input_shape, ref_shape)
        array_c = deepcopy(array_s)
        self.assertEqual(array_c.input_shape, ref_shape)

        array_c = array.deepcopy()
        self.assertEqual(array_c.input_shape, ref_shape)
        array_c = array_s.deepcopy()
        self.assertEqual(array_c.input_shape, ref_shape)

        # should propagate on setitem
        array_c = array.deepcopy()
        array_c[0:2, 0:2] = 200
        self.assertEqual(array_c.input_shape, ref_shape)

        array_c = array.deepcopy()
        array_c[0:2, 0:2] = CArray([[100, 200], [300, 400]])
        self.assertEqual(array_c.input_shape, ref_shape)

        array_c = array_s.deepcopy()
        array_c[0:2, 0:2] = CArray([[100, 200], [300, 400]])
        self.assertEqual(array_c.input_shape, ref_shape)

        # should propagate on todense/tosparse
        self.assertEqual(array.tosparse().input_shape, ref_shape)
        self.assertEqual(array.todense().input_shape, ref_shape)
        self.assertEqual(array_s.tosparse().input_shape, ref_shape)
        self.assertEqual(array_s.todense().input_shape, ref_shape)


if __name__ == '__main__':
    CArrayTestCases.main()
