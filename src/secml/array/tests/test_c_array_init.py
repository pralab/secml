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

    def test_init(self):
        """Test CArray initialization"""

        def test_init_builtin(totest_list):

            for totest_elem in totest_list:
                for tosparse in [False, True]:
                    init_array = CArray(totest_elem, tosparse=tosparse)
                    self.assertTrue(init_array.issparse == tosparse)
                    if is_list_of_lists(totest_elem):
                        self.assertTrue(
                            init_array.shape[0] == len(totest_elem))
                        self.assertTrue(
                            init_array.shape[1] == len(totest_elem[0]))
                    elif is_list(totest_elem):
                        if init_array.issparse is True:
                            self.assertTrue(
                                init_array.shape[1] == len(totest_elem))
                        elif init_array.isdense is True:
                            self.assertTrue(init_array.ndim == 1)
                            self.assertTrue(
                                init_array.shape[0] == len(totest_elem))
                    elif is_scalar(totest_elem) or is_bool(totest_elem):
                        self.assertTrue(init_array.size == 1)
                    else:
                        raise TypeError(
                            "test_init_builtin should not be used "
                            "to test {:}".format(type(totest_elem)))

        self.logger.info("Initializing CArray with built-in types...")
        test_init_builtin([[[2, 3], [22, 33]], [2, 3], [[2], [3]], 3, True,
                           [[True, False], [True, True]], [True, False]])

        self.logger.info("Initializing CArray with another CArray...")
        arrays_list = [CArray([[2, 3], [22, 33]]),
                      CArray([2, 3]),
                      CArray([[2], [3]]),
                      CArray(3)]

        for init_elem in arrays_list:
            array = CArray(init_elem)
            self.assertTrue(init_elem.issparse == array.issparse)
            self.assertFalse((array != init_elem).any())

        self.logger.info(
            "Initializing CArray with a CDense or an ndarray...")
        dense_list = [CDense([[2, 3], [22, 33]]),
                      CDense([2, 3]),
                      CDense([[2], [3]]),
                      CDense([3]),
                      np.array([[2, 3], [22, 33]]),
                      np.array([2, 3]),
                      np.array([[2], [3]]),
                      np.array([3])]

        for init_elem in dense_list:
            array = CArray(init_elem)
            self.assertTrue(array.isdense)
            self.assertTrue(array.shape == init_elem.shape)

        self.logger.info("Initializing CArray with a sparse CArray...")
        sparse_list = [CArray([[2, 3], [22, 33]], tosparse=True),
                       CArray([2, 3], tosparse=True),
                       CArray([[2], [3]], tosparse=True),
                       CArray(3, tosparse=True)]

        for init_elem in sparse_list:
            array = CArray(init_elem)
            self.assertTrue(array.issparse)
            self.assertFalse((array != init_elem).any())

        self.logger.info(
            "Initializing CArray with a CSparse or csr_matrix...")
        sparse_list = [CSparse([[2, 3], [22, 33]]),
                       CSparse([2, 3]),
                       CSparse([[2], [3]]),
                       CSparse([3]),
                       scs.csr_matrix([[2, 3], [22, 33]]),
                       scs.csr_matrix([2, 3]),
                       scs.csr_matrix([[2], [3]]),
                       scs.csr_matrix([3])]

        for init_elem in sparse_list:
            array = CArray(init_elem)
            self.assertTrue(array.issparse)
            self.assertTrue(array.shape == init_elem.shape)

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


if __name__ == '__main__':
    CArrayTestCases.main()
