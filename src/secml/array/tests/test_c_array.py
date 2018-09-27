import numpy as np
import scipy.sparse as scs
import operator as op
import itertools

from secml.utils import CUnitTest, fm
from secml.array import CArray, Cdense, Csparse
from secml.core.type_utils import \
    is_scalar, is_int, is_bool, is_list, is_list_of_lists

from numpy import matlib

import copy 


class TestCArray(CUnitTest):
    """Unit test for CArray."""
     
    def setUp(self):
        """Basic set up."""
        self.array_dense = CArray([[1, 0, 0, 5],
                                   [2, 4, 0, 0],
                                   [3, 6, 0, 0]])
        self.array_sparse = CArray(
            self.array_dense.deepcopy(), tosparse=True)

        self.array_dense_sym = CArray([[1, 2, 0],
                                       [2, 4, 6],
                                       [0, 6, 0]])
        self.array_sparse_sym = CArray(
            self.array_dense_sym.deepcopy(), tosparse=True)

        self.array_dense_nozero = CArray([[1, 2, 3, 4],
                                          [5, 6, 7, 8],
                                          [9, 10, 11, 12]])
        self.array_sparse_nozero = CArray(
            self.array_dense_nozero.deepcopy(), tosparse=True)

        self.array_dense_allzero = CArray([[0, 0, 0, 0],
                                           [0, 0, 0, 0],
                                           [0, 0, 0, 0]])
        self.array_sparse_allzero = CArray(
            self.array_dense_allzero.deepcopy(), tosparse=True)

        self.array_dense_bool = CArray([[True, False, True, True],
                                        [False, False, False, False],
                                        [True, True, True, True]])
        self.array_sparse_bool = CArray(
            self.array_dense_bool.deepcopy(), tosparse=True)

        self.array_dense_bool_true = CArray([[True, True, True, True],
                                             [True, True, True, True],
                                             [True, True, True, True]])
        self.array_sparse_bool_true = CArray(
            self.array_dense_bool_true.deepcopy(), tosparse=True)

        self.array_dense_bool_false = CArray([[False, False, False, False],
                                              [False, False, False, False],
                                              [False, False, False, False]])
        self.array_sparse_bool_false = CArray(
            self.array_dense_bool_false.deepcopy(), tosparse=True)

        self.row_flat_dense = CArray([4, 0, 6])
        self.row_dense = self.row_flat_dense.atleast_2d()
        self.column_dense = self.row_dense.deepcopy().T

        self.row_sparse = CArray(self.row_dense.deepcopy(), tosparse=True)
        self.column_sparse = self.row_sparse.deepcopy().T

        self.single_flat_dense = CArray([4])
        self.single_dense = self.single_flat_dense.atleast_2d()
        self.single_sparse = CArray(
            self.single_dense.deepcopy(), tosparse=True)

        self.single_bool_flat_dense = CArray([True])
        self.single_bool_dense = self.single_bool_flat_dense.atleast_2d()
        self.single_bool_sparse = CArray(
            self.single_bool_dense.deepcopy(), tosparse=True)

        self.empty_flat_dense = CArray([], tosparse=False)
        self.empty_dense = CArray([[]], tosparse=False)
        self.empty_sparse = CArray([], tosparse=True)

    def _test_multiple_eq(self, items_list):
        """Return True if all items are equal."""

        # We are going to compare the first element
        # with the second, the second with the third, etc.
        for item_idx, item in enumerate(items_list):
            if item_idx == len(items_list) - 1:
                break  # We checked all the elements
            np.testing.assert_equal(item.tondarray(),
                                    items_list[item_idx + 1].tondarray())

        # Every item is equal to each other, return True
        return True

    def _test_cycle(self, totest_op, totest_items, totest_result):
        """Check if operator return the expected result on given items.

        totest_op: list of operators
        totest_items: list of items PAIR to test
        totest_result: list of expected result (class name) for each PAIR

        """
        for operator in totest_op:
            to_check = []
            for pair_idx, pair in enumerate(totest_items):
                class0 = type(pair[0]._data) if \
                    hasattr(pair[0], 'isdense') else type(pair[0])
                class1 = type(pair[1]._data) if \
                    hasattr(pair[1], 'isdense') else type(pair[1])
                self.logger.info("Operator {:} between {:} and {:}"
                                 "".format(operator.__name__, class0, class1))
                result = operator(pair[0], pair[1])
                self.assertIsInstance(result._data, totest_result[pair_idx])
                self.logger.info(
                    "Result: {:}".format(result._data.__class__.__name__))
                to_check.append(result)
            self.assertTrue(self._test_multiple_eq(to_check))

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
            "Initializing CArray with a Cdense or an ndarray...")
        dense_list = [Cdense([[2, 3], [22, 33]]),
                       Cdense([2, 3]),
                       Cdense([[2], [3]]),
                       Cdense([3]),
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
            "Initializing CArray with a Csparse or csr_matrix...")
        sparse_list = [Csparse([[2, 3], [22, 33]]),
                       Csparse([2, 3]),
                       Csparse([[2], [3]]),
                       Csparse([3]),
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

    def test_getter(self):
        """Method that tests __getitem__ methods."""

        def test_selectors(input_array, selector_list, target_list):

            for selector_idx, selector in enumerate(selector_list):

                self.logger.info("Get: array[{:}]".format(selector))
                try:  # Using a try to easier debug
                    selection = input_array[selector]
                except (IndexError, ValueError, TypeError):
                    selection = input_array[selector]
                self.logger.info("Result is: \n" + str(selection))

                self.assertFalse(CArray(selection != target_list[selector_idx]).any(),
                                 "{:} is different from {:}".format(selection, target_list[selector_idx]))

                if isinstance(target_list[selector_idx], CArray):
                    if selection.issparse:
                        self.assertEqual(selection.shape,
                                         target_list[
                                             selector_idx].atleast_2d().shape)
                    else:
                        self.assertEqual(selection.shape,
                                         target_list[selector_idx].shape)

        # 2D/1D INDEXING (MATRIX)
        arrays_list = [self.array_dense, self.array_sparse]
        for array in arrays_list:

            self.logger.info("Testing getters for matrix: \n" + str(array))

            selectors = [[[1, 2, 2, 2], [2, 0, 1, 2]],
                         [[1, 2, 2, 2], [np.ravel(2)[0], np.ravel(0)[0], np.ravel(1)[0], np.ravel(2)[0]]],
                         [[np.ravel(1)[0], np.ravel(2)[0], np.ravel(2)[0], np.ravel(2)[0]], [2, 0, 1, 2]],
                         [[np.ravel(1)[0], np.ravel(2)[0], np.ravel(2)[0], np.ravel(2)[0]],
                          [np.ravel(2)[0], np.ravel(0)[0], np.ravel(1)[0], np.ravel(2)[0]]],
                         CArray([[True, False, True, False],
                                 [False, False, False, False],
                                 [False, True, False, False]]),
                         ]
            targets = [CArray([0, 3, 6, 0]),
                       CArray([0, 3, 6, 0]),
                       CArray([0, 3, 6, 0]),
                       CArray([0, 3, 6, 0]),
                       CArray([1, 0, 6])
                       ]

            test_selectors(array, selectors, targets)

        # 2D/1D INDEXING (MATRIX SYMMETRIC, for easier testing of different indices)
        arrays_list = [self.array_dense_sym, self.array_sparse_sym]
        for array in arrays_list:

            self.logger.info("Testing getters for matrix: \n" + str(array))

            selectors_unique = [2, np.ravel(2)[0], [2, 2], CArray([2, 2]),
                                slice(1, 3), [False, True, True], CArray([False, True, True])]
            selectors = itertools.product(selectors_unique, repeat=2)

            targets_a = [0, 0, CArray([[0, 0]]), CArray([[0, 0]]),  # 2
                         CArray([[6, 0]]), CArray([[6, 0]]), CArray([[6, 0]])
                         ]
            targets_b = [CArray([[0], [0]]), CArray([[0], [0]]), CArray([[0, 0], [0, 0]]),  # [2, 2]
                         CArray([[0, 0], [0, 0]]), CArray([[6, 0], [6, 0]]),
                         CArray([[6, 0], [6, 0]]), CArray([[6, 0], [6, 0]])
                         ]
            targets_c = [CArray([[6], [0]]), CArray([[6], [0]]), CArray([[6, 6], [0, 0]]),  # [False, True, True]
                         CArray([[6, 6], [0, 0]]), CArray([[4, 6], [6, 0]]),
                         CArray([[4, 6], [6, 0]]), CArray([[4, 6], [6, 0]])
                         ]

            targets = 2 * targets_a + 2 * targets_b + 3 * targets_c

            test_selectors(array, selectors, targets)

        # 2D/1D INDEXING (VECTOR-LIKE)
        arrays_list = [self.row_flat_dense, self.row_dense, self.row_sparse]
        for array in arrays_list:

            self.logger.info("Testing getters for array: \n" + str(array))

            selectors_a = [[[0, 0, 0], [2, 0, 1]],
                           [[0, 0, 0], [np.ravel(2)[0], np.ravel(0)[0], np.ravel(1)[0]]],
                           [[np.ravel(0)[0], np.ravel(0)[0], np.ravel(0)[0]], [2, 0, 1]],
                           [[np.ravel(0)[0], np.ravel(0)[0], np.ravel(0)[0]],
                            [np.ravel(2)[0], np.ravel(0)[0], np.ravel(1)[0]]],
                           CArray([[True, False, True]]),
                           CArray([True, False, True])
                           ]
            selectors_row = [0, np.ravel(0)[0], [0], CArray([0]),
                             -1, np.ravel(-1)[0], [-1], CArray([-1]),
                             True, np.ravel(True)[0], [True], CArray([True])]
            selectors_col = [0, np.ravel(0)[0], [2, 2], CArray([2, 2]),
                             slice(1, 3), [False, True, True], CArray([False, True, True])]
            selectors = selectors_a + [(x, y) for x in selectors_row for y in selectors_col]

            targets_a = [CArray([6, 4, 0])]
            targets_b = [CArray([4, 6])]
            targets = 4 * targets_a + 2 * targets_b
            # Output always flat for flat arrays
            if array.ndim == 1:
                targets += 12 * (2 * [4] + 2 * [CArray([6, 6])] + 3 * [CArray([0, 6])])
            else:
                targets += 12 * (2 * [4] + 2 * [CArray([[6, 6]])] + 3 * [CArray([[0, 6]])])

            test_selectors(array, selectors, targets)

        # 1D INDEXING (VECTOR-LIKE)
        arrays_list = [self.row_flat_dense, self.row_dense, self.row_sparse]
        for array in arrays_list:

            self.logger.info("Testing getters for vector: \n" + str(array))

            selectors = [0, np.ravel(0)[0], [2, 2], CArray([2, 2]), slice(1, 3), slice(None)]

            # Output always flat for flat arrays
            if array.ndim == 1:
                targets = 2 * [4] + 2 * [CArray([6, 6])] + [CArray([0, 6])] + [CArray([4, 0, 6])]
            else:
                targets = 2 * [4] + 2 * [CArray([[6, 6]])] + [CArray([[0, 6]])] + [CArray([[4, 0, 6]])]

            test_selectors(array, selectors, targets)

        # SPECIAL CASE: SIZE 1 ARRAY
        arrays_list = [self.single_flat_dense, self.single_dense, self.single_sparse]
        for array in arrays_list:

            self.logger.info("Testing getters for array: \n" + str(array))

            selectors = [0, np.ravel(0)[0], True, [True], CArray([True]), slice(0, 1), slice(None), CArray([0, 0])]

            targets = 7 * [4]
            # Output always flat for flat arrays
            targets += [CArray([4, 4])] if array.ndim == 1 else [CArray([[4, 4]])]

            test_selectors(array, selectors, targets)

    def test_setter(self):
        """Method that tests __setitem__ methods."""

        def test_selectors(input_array, selector_list, assignment_list, target_list):

            for selector_idx, selector in enumerate(selector_list):

                self.logger.info("Set: array[{:}] = {:}".format(selector, assignment_list[selector_idx]))
                array_copy = input_array.deepcopy()
                try:  # Using a try to easier debug
                    array_copy[selector] = assignment_list[selector_idx]
                except (IndexError, ValueError, TypeError):
                    array_copy[selector] = assignment_list[selector_idx]
                self.logger.info("Result is: \n" + str(array_copy))

                self.assertFalse(CArray(array_copy != target_list[selector_idx]).any(),
                                 "{:} is different from {:}".format(array_copy, target_list[selector_idx]))

                if hasattr(target_list[selector_idx], 'shape'):
                    self.assertEqual(array_copy.shape, target_list[selector_idx].shape)

        # 2D/1D INDEXING (MATRIX)
        arrays_list = [self.array_dense, self.array_sparse]
        for array in arrays_list:

            self.logger.info("Testing setters for matrix: \n" + str(array))

            selectors = [[[1, 2, 2, 2], [2, 0, 1, 2]],
                         [[1, 2, 2, 2], [np.ravel(2)[0], np.ravel(0)[0], np.ravel(1)[0], np.ravel(2)[0]]],
                         [[np.ravel(1)[0], np.ravel(2)[0], np.ravel(2)[0], np.ravel(2)[0]], [2, 0, 1, 2]],
                         [[np.ravel(1)[0], np.ravel(2)[0], np.ravel(2)[0], np.ravel(2)[0]],
                          [np.ravel(2)[0], np.ravel(0)[0], np.ravel(1)[0], np.ravel(2)[0]]]
                         ]
            selectors += 3 * [CArray([[False, False, False, False],
                                      [False, False, True, False],
                                      [True, True, True, False]])]

            assignments = [10, 10, CArray([10, 20, 30, 40]), CArray([10, 20, 30, 40]),
                           CArray([10, 20, 30, 40]), CArray([10, 20, 30, 40]), 10
                           ]

            targets_a = [CArray([[1, 0, 0, 5], [2, 4, 10, 0], [10, 10, 10, 0]])]
            targets_b = [CArray([[1, 0, 0, 5], [2, 4, 10, 0], [20, 30, 40, 0]])]
            targets = 2 * targets_a + 4 * targets_b + targets_a

            test_selectors(array, selectors, assignments, targets)

        # 2D/1D INDEXING (MATRIX SYMMETRIC, for easier testing of different indices)
        arrays_list = [self.array_dense_sym, self.array_sparse_sym]
        for array in arrays_list:

            self.logger.info("Testing setters for matrix: \n" + str(array))

            selectors_unique = [2, np.ravel(2)[0], [1, 2], CArray([1, 2]),
                                slice(1, 3), [False, True, True], CArray([False, True, True])]
            selectors = itertools.product(selectors_unique, repeat=2)

            assignments_a = [10, 10] + 3 * [CArray([[10, 20]])] + 2 * [CArray([10, 20])]
            assignments_b = [CArray([[10], [20]])] + [CArray([[10], [20]], tosparse=True)] + \
                            5 * [CArray([[10, 20], [30, 40]])]
            assignments = 2 * assignments_a + 5 * assignments_b

            targets_a = 2 * [CArray([[1, 2, 0], [2, 4, 6], [0, 6, 10]])] + \
                        5 * [CArray([[1, 2, 0], [2, 4, 6], [0, 10, 20]])]
            targets_b = 2 * [CArray([[1, 2, 0], [2, 4, 10], [0, 6, 20]])] + \
                        5 * [CArray([[1, 2, 0], [2, 10, 20], [0, 30, 40]])]
            targets = 2 * targets_a + 5 * targets_b

            test_selectors(array, selectors, assignments, targets)

        # 2D/1D INDEXING (VECTOR-LIKE)
        arrays_list = [self.row_flat_dense, self.row_dense, self.row_sparse]
        for array in arrays_list:

            self.logger.info("Testing setters for array: \n" + str(array))

            selectors_a = [[[0, 0], [2, 0]],
                           [[0, 0], [np.ravel(2)[0], np.ravel(0)[0]]],
                           [[np.ravel(0)[0], np.ravel(0)[0]], [2, 0]],
                           [[np.ravel(0)[0], np.ravel(0)[0]], [np.ravel(2)[0], np.ravel(0)[0]]],
                           CArray([[True, False, True]]),
                           CArray([True, False, True])
                           ]
            selectors_row = [0, np.ravel(0)[0], [0], CArray([0]),
                             -1, np.ravel(-1)[0], [-1], CArray([-1]),
                             True, np.ravel(True)[0], [True], CArray([True])]
            selectors_col = [0, np.ravel(0)[0], [1, 2], CArray([1, 2]),
                             slice(1, 3), [False, True, True], CArray([False, True, True])]
            selectors = selectors_a + [(x, y) for x in selectors_row for y in selectors_col]

            assignments_a = 2 * [CArray([10, 20])] + 2 * [CArray([[10, 20]])] + 2 * [CArray([10, 20])]
            assignments_b = [10, 10] + 2 * [CArray([[10, 20]])] + 3 * [CArray([10, 20])]
            assignments = assignments_a + 12 * assignments_b

            targets_a = CArray([20, 0, 10])
            targets_b = CArray([10, 0, 20])
            targets_c = CArray([10, 0, 6])
            targets_d = CArray([4, 10, 20])
            # Output always flat for flat arrays
            if array.ndim == 1:
                targets = 4 * [targets_a] + 2 * [targets_b] + 12 * (2 * [targets_c] + 5 * [targets_d])
            else:
                targets = 4 * [targets_a.atleast_2d()] + 2 * [targets_b.atleast_2d()] + \
                            12 * (2 * [targets_c.atleast_2d()] + 5 * [targets_d.atleast_2d()])

            test_selectors(array, selectors, assignments, targets)

        # 1D INDEXING (VECTOR-LIKE)
        arrays_list = [self.row_flat_dense, self.row_dense, self.row_sparse]
        for array in arrays_list:

            self.logger.info("Testing setters for vector: \n" + str(array))

            selectors = [0, np.ravel(0)[0], [1, 2], CArray([1, 2]), slice(1, 3), slice(None)]

            assignments = [10, 10] + 2 * [CArray([[10, 20]])] + \
                          [CArray([[10, 20]], tosparse=True)] + [CArray([[10, 20, 30]])]

            targets_a = CArray([10, 0, 6])
            targets_b = CArray([4, 10, 20])
            targets_c = CArray([10, 20, 30])
            # Output always flat for flat arrays
            if array.ndim == 1:
                targets = 2 * [targets_a] + 3 * [targets_b] + [targets_c]
            else:
                targets = 2 * [targets_a.atleast_2d()] + 3 * [targets_b.atleast_2d()] + [targets_c.atleast_2d()]

            test_selectors(array, selectors, assignments, targets)

        # SPECIAL CASE: SIZE 1 ARRAY
        arrays_list = [self.single_flat_dense, self.single_dense, self.single_sparse]
        for array in arrays_list:

            self.logger.info("Testing setters for array: \n" + str(array))

            selectors = [0, np.ravel(0)[0], True, [True], CArray([True]), slice(0, 1), slice(None)]

            assignments = 7 * [10]

            targets_a = CArray([10])
            # Output always flat for flat arrays
            if array.ndim == 1:
                targets = 7 * [targets_a]
            else:
                targets = 7 * [targets_a.atleast_2d()]

            test_selectors(array, selectors, assignments, targets)

    def test_all(self):
        """Test for CArray.all() method."""
        self.logger.info("Test for CArray.all() method")
                 
        def _all(matrix, matrix_nozero, matrix_bool, matrix_bool_true):
             
            # all() on an array that contain also zeros gives False?
            self.logger.info("matrix: \n" + str(matrix))
            all_res = matrix.all()
            self.logger.info("matrix.all() result is: \n" + str(all_res))
            self.assertFalse(all_res)
 
            # all() on an array with no zeros gives True?
            self.logger.info("matrix_nozero: \n" + str(matrix_nozero))
            all_res = matrix_nozero.all()
            self.logger.info("matrix_nozero.all(): \n" + str(all_res))
            self.assertTrue(all_res)

            # all() on boolean array
            self.logger.info("matrix_bool: \n" + str(matrix_bool))
            all_res = matrix_bool.all()
            self.logger.info("matrix_bool.all(): \n" + str(all_res))
            self.assertFalse(all_res)
 
            # all() on a boolean array with all True
            self.logger.info("matrix_bool_true: \n" + str(matrix_bool_true))
            all_res = matrix_bool_true.all()
            self.logger.info("matrix_bool_true.all(): \n" + str(all_res))
            self.assertTrue(all_res)

        _all(self.array_sparse, self.array_sparse_nozero,
             self.array_sparse_bool, self.array_sparse_bool_true)
        _all(self.array_dense, self.array_dense_nozero,
             self.array_dense_bool, self.array_dense_bool_true)

    def test_any(self):
        """Test for CArray.any() method."""
        self.logger.info("Test for CArray.any() method")

        def _any(matrix, matrix_allzero, matrix_bool, matrix_bool_false):

            # any() on an array that contain also zeros gives True?
            self.logger.info("matrix: \n" + str(matrix))
            any_res = matrix.any()
            self.logger.info("matrix.any() result is: \n" + str(any_res))
            self.assertTrue(any_res)

            # any() on an array with all zeros gives False?
            self.logger.info("matrix_allzero: \n" + str(matrix_allzero))
            any_res = matrix_allzero.any()
            self.logger.info("matrix_allzero.any(): \n" + str(any_res))
            self.assertFalse(any_res)

            # any() on boolean array
            self.logger.info("matrix_bool: \n" + str(matrix_bool))
            any_res = matrix_bool.any()
            self.logger.info("matrix_bool.any(): \n" + str(any_res))
            self.assertTrue(any_res)

            # any() on a boolean array with all False
            self.logger.info("matrix_bool_false: \n" + str(matrix_bool_false))
            any_res = matrix_bool_false.any()
            self.logger.info("matrix_bool_false.any(): \n" + str(any_res))
            self.assertFalse(any_res)

        _any(self.array_sparse, self.array_sparse_allzero,
             self.array_sparse_bool, self.array_sparse_bool_false)
        _any(self.array_dense, self.array_dense_allzero,
             self.array_dense_bool, self.array_dense_bool_false)

    def test_deepcopy(self):
        """Test for CArray.deepcopy() method."""
        self.logger.info("Test for CArray.deepcopy() method")

        def _deepcopy(array):

            self.logger.info("Array:\n{:}".format(array))

            array_deepcopy = array.deepcopy()
            self.logger.info("Array deepcopied:\n{:}".format(
                array_deepcopy.todense()))

            self.assertEquals(array.issparse, array_deepcopy.issparse)
            self.assertEquals(array.isdense, array_deepcopy.isdense)

            # copy method must return a copy of data
            array_deepcopy[:, :] = 9
            self.assertFalse((array_deepcopy == array).any())

        _deepcopy(self.array_sparse)
        _deepcopy(self.array_dense)

    def test_unique(self):
        """Test for CArray.unique() method."""
        self.logger.info("Test for CArray.unique() method")
    
        def _unique(array, true_unique):

            self.logger.info("Array:\n{:}".format(array))

            if array.isdense:
                array_unique, unique_indices, unique_inverse = array.unique(
                    return_index=True, return_inverse=True)
                # Testing call without the optional parameters
                array_unique_single = array.unique()
                self.assertFalse((array_unique != array_unique_single).any())
            elif array.issparse:
                # return_index, return_inverse parameters are not available
                # for sparse arrays
                with self.assertRaises(ValueError):
                    array.unique(return_index=True, return_inverse=True)
                array_unique = array.unique()
            else:
                raise ValueError("Unknown input array format")
            self.logger.info("array.unique():\n{:}".format(array_unique))

            self.assertIsInstance(array_unique, CArray)
            # output of unique method must be dense
            self.assertTrue(array_unique.isdense)

            self.assertEqual(true_unique.size, array_unique.size)

            unique_ok = True
            for num in true_unique:
                if num not in array_unique:
                    unique_ok = False
            self.assertTrue(unique_ok)

            if array.isdense:
                # unique_inverse reconstruct the original FLAT array
                self.assertFalse(
                    (array.ravel() != array_unique[unique_inverse]).any())
                # unique_indices construct unique array from original FLAT one
                self.assertFalse(
                    (array.ravel()[unique_indices] != array_unique).any())

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

    def test_diag(self):
        """Test for CArray.diag() method."""
        self.logger.info("Test for CArray.diag() method.")

        def extract_diag(array, k, out):

            diag = array.diag(k=k)
            self.logger.info("({:})-th diagonal is: {:}".format(k, diag))
            self.assertEquals(diag.ndim, 1)
            self.assertTrue((diag == out).all())

        self.logger.info("Testing diagonal extraction...")

        self.logger.info("Array is:\n{:}".format(self.array_dense))

        extract_diag(self.array_dense, k=0, out=CArray([1, 4, 0]))
        extract_diag(self.array_dense, k=1, out=CArray([0, 0, 0]))
        extract_diag(self.array_dense, k=-1, out=CArray([2, 6]))
        extract_diag(self.array_dense, k=5, out=CArray([]))

        self.logger.info("Array is:\n{:}".format(self.array_sparse))

        diag_sparse = self.array_sparse.diag()
        self.logger.info("0-th diagonal is: {:}".format(diag_sparse))
        self.assertTrue(diag_sparse.issparse)
        self.assertEquals(diag_sparse.ndim, 2)
        self.assertTrue((diag_sparse == CArray([1, 4, 0])).all())

        with self.assertRaises(ValueError):
            self.array_sparse.diag(k=1)

        self.logger.info("Testing diagonal array creation...")

        def create_diag(array, k, out):

            diag = array.diag(k=k)
            self.logger.info(
                "Array created using k={:} is:\n{:}".format(k, diag))
            self.assertEquals(array.isdense, diag.isdense)
            self.assertEquals(array.issparse, diag.issparse)
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

            self.assertEquals(array.shape, array_shape)
            self.assertEquals(array.issparse, array_issparse)
            self.assertEquals(array.isdense, array_isdense)

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

    def test_sort(self):
        """Test for CArray.sort() method."""
        self.logger.info("Test for CArray.sort() method")

        def _sort(axis, array, sorted_expected):
            self.logger.info("Array:\n{:}".format(array))

            array_isdense = array.isdense
            array_issparse = array.issparse

            array_sorted = copy.deepcopy(array)  # in-place method
            array_sorted.sort(axis=axis)
            self.logger.info(
                "Array sorted along axis {:}:\n{:}".format(axis, array_sorted))

            self.assertEquals(array_sorted.issparse, array_issparse)
            self.assertEquals(array_sorted.isdense, array_isdense)

            self.assertFalse((sorted_expected != array_sorted).any())

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

    def test_argmin(self):
        """Test for CArray.argmin() method."""
        self.logger.info("Test for CArray.argmin() method.")

        def _argmin(array):

            self.logger.info("a: \n{:}".format(array))
            argmin_res = array.argmin(axis=None)
            self.logger.info("a.argmin(axis=None): \n{:}".format(argmin_res))
            self.assertIsInstance(argmin_res, int)
            min_res = array.min(axis=None)
            self.assertEqual(array.ravel()[argmin_res], min_res)

            self.logger.info("a: \n{:}".format(array))
            argmin_res = array.argmin(axis=0)
            self.logger.info("a.argmin(axis=0): \n{:}".format(argmin_res))
            min_res = array.min(axis=0)
            if array.shape[1] > 1:  # One res for each column with keepdims
                min_res = min_res.ravel()
                # We create a find_2d-like mask to check result
                argmin_res = [
                    argmin_res.ravel().tolist(), range(array.shape[1])]
                self.assertTrue((array[argmin_res] == min_res).all())
            else:  # Result is a scalar
                self.assertEqual(array.ravel()[argmin_res], min_res)

            self.logger.info("a: \n{:}".format(array))
            argmin_res = array.argmin(axis=1)
            self.logger.info("a.argmin(axis=1): \n{:}".format(argmin_res))
            min_res = array.min(axis=1)
            if array.shape[0] > 1:  # One res for each row with keepdims
                min_res = min_res.ravel()
                # We create a find_2d-like mask to check result
                argmin_res = [
                    range(array.shape[0]), argmin_res.ravel().tolist()]
                self.assertTrue((array[argmin_res] == min_res).all())
            else:  # Result is a scalar
                self.assertEqual(array.ravel()[argmin_res], min_res)
                
        _argmin(self.array_sparse)
        _argmin(self.row_sparse)
        _argmin(self.column_sparse)
        _argmin(self.array_dense)
        _argmin(self.row_sparse)
        _argmin(self.column_dense)

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
            self.assertEqual(array.ravel()[argmax_res], max_res)

            self.logger.info("a: \n{:}".format(array))
            argmax_res = array.argmax(axis=0)
            self.logger.info("a.argmax(axis=0): \n{:}".format(argmax_res))
            max_res = array.max(axis=0)
            if array.shape[1] > 1:  # One res for each column with keepdims
                max_res = max_res.ravel()
                # We create a find_2d-like mask to check result
                argmax_res = [argmax_res.ravel().tolist(), range(array.shape[1])]
                self.assertTrue((array[argmax_res] == max_res).all())
            else:  # Result is a scalar
                self.assertEqual(array.ravel()[argmax_res], max_res)

            self.logger.info("a: \n{:}".format(array))
            argmax_res = array.argmax(axis=1)
            self.logger.info("a.argmax(axis=1): \n{:}".format(argmax_res))
            max_res = array.max(axis=1)
            if array.shape[0] > 1:  # One res for each row with keepdims
                max_res = max_res.ravel()
                # We create a find_2d-like mask to check result
                argmax_res = [
                    range(array.shape[0]), argmax_res.ravel().tolist()]
                self.assertTrue((array[argmax_res] == max_res).all())
            else:  # Result is a scalar
                self.assertEqual(array.ravel()[argmax_res], max_res)

        _argmax(self.array_sparse)
        _argmax(self.row_sparse)
        _argmax(self.column_sparse)
        _argmax(self.array_dense)
        _argmax(self.row_sparse)
        _argmax(self.column_dense)

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
            array[0, 0] = np.nan

            self.logger.info("a: \n{:}".format(array))
            argmin_res = array.nanargmin(axis=None)
            self.logger.info(
                "a.nanargmin(axis=None): \n{:}".format(argmin_res))
            self.assertIsInstance(argmin_res, int)
            min_res = array.nanmin(axis=None)
            # use numpy.testing to proper compare arrays with nans
            np.testing.assert_equal(
                CArray(array.ravel()[argmin_res]).tondarray(),
                CArray(min_res).tondarray())

            self.logger.info("a: \n{:}".format(array))
            if array.shape[0] == 1:
                # ValueError: All-NaN slice encountered
                with self.assertRaises(ValueError):
                    array.nanargmin(axis=0)
            else:
                argmin_res = array.nanargmin(axis=0)
                self.logger.info(
                    "a.nanargmin(axis=0): \n{:}".format(argmin_res))
                min_res = array.nanmin(axis=0)
                if array.shape[1] > 1:  # One res for each column with keepdims
                    min_res = min_res.ravel()
                    argmin_res = [
                        argmin_res.ravel().tolist(), range(array.shape[1])]
                    # use numpy.testing to proper compare arrays with nans
                    np.testing.assert_equal(
                        CArray(array[argmin_res]).tondarray(),
                        CArray(min_res).tondarray())
                else:  # Result is a scalar
                    np.testing.assert_equal(
                        CArray(array.ravel()[argmin_res]).tondarray(),
                        CArray(min_res).tondarray())

            self.logger.info("a: \n{:}".format(array))
            if array.shape[1] == 1:
                # ValueError: All-NaN slice encountered
                with self.assertRaises(ValueError):
                    array.nanargmin(axis=1)
            else:
                argmin_res = array.nanargmin(axis=1)
                self.logger.info(
                    "a.nanargmin(axis=1): \n{:}".format(argmin_res))
                min_res = array.nanmin(axis=1)
                if array.shape[0] > 1:  # One res for each row with keepdims
                    min_res = min_res.ravel()
                    argmin_res = [
                        range(array.shape[0]), argmin_res.ravel().tolist()]
                    # use numpy.testing to proper compare arrays with nans
                    np.testing.assert_equal(
                        CArray(array[argmin_res]).tondarray(),
                        CArray(min_res).tondarray())
                else:  # Result is a scalar
                    np.testing.assert_equal(
                        CArray(array.ravel()[argmin_res]).tondarray(),
                        CArray(min_res).tondarray())

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
            array[0, 0] = np.nan

            self.logger.info("a: \n{:}".format(array))
            argmax_res = array.nanargmax(axis=None)
            self.logger.info(
                "a.nanargmax(axis=None): \n{:}".format(argmax_res))
            self.assertIsInstance(argmax_res, int)
            max_res = array.nanmax(axis=None)
            np.testing.assert_equal(
                CArray(array.ravel()[argmax_res]).tondarray(),
                CArray(max_res).tondarray())

            self.logger.info("a: \n{:}".format(array))
            if array.shape[0] == 1:
                # ValueError: All-NaN slice encountered
                with self.assertRaises(ValueError):
                    array.nanargmax(axis=0)
            else:
                argmax_res = array.nanargmax(axis=0)
                self.logger.info(
                    "a.nanargmax(axis=0): \n{:}".format(argmax_res))
                max_res = array.nanmax(axis=0)
                if array.shape[1] > 1:  # One res for each column with keepdims
                    max_res = max_res.ravel()
                    argmax_res = [
                        argmax_res.ravel().tolist(), range(array.shape[1])]
                    np.testing.assert_equal(
                        CArray(array[argmax_res]).tondarray(),
                        CArray(max_res).tondarray())
                else:  # Result is a scalar
                    np.testing.assert_equal(
                        CArray(array.ravel()[argmax_res]).tondarray(),
                        CArray(max_res).tondarray())

            self.logger.info("a: \n{:}".format(array))
            if array.shape[1] == 1:
                # ValueError: All-NaN slice encountered
                with self.assertRaises(ValueError):
                    array.nanargmax(axis=1)
            else:
                argmax_res = array.nanargmax(axis=1)
                self.logger.info(
                    "a.nanargmax(axis=1): \n{:}".format(argmax_res))
                max_res = array.nanmax(axis=1)
                if array.shape[0] > 1:  # One res for each row with keepdims
                    max_res = max_res.ravel()
                    argmax_res = [
                        range(array.shape[0]), argmax_res.ravel().tolist()]
                    np.testing.assert_equal(
                        CArray(array[argmax_res]).tondarray(),
                        CArray(max_res).tondarray())
                else:  # Result is a scalar
                    np.testing.assert_equal(
                        CArray(array.ravel()[argmax_res]).tondarray(),
                        CArray(max_res).tondarray())

        _check_nanargmax(self.array_dense)
        _check_nanargmax(self.row_dense)
        _check_nanargmax(self.column_dense)

        # nanargmax on empty arrays should raise ValueError
        with self.assertRaises(ValueError):
            self.empty_dense.nanargmax()

        with self.assertRaises(NotImplementedError):
            self.array_sparse.nanargmax()

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

            self.assertTrue((maximum_res == array2).all())

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

            self.assertTrue((minimum_res == array1).all())

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

                    res_expected = expected[res_idx]
                    if not isinstance(res_expected, CArray):
                        self.assertNotIsInstance(res, CArray)
                        res = CArray(res).round(2)[0]
                        self.assertEqual(res, res_expected)
                    else:
                        self.assertIsInstance(res, CArray)
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

        _check_median(self.row_flat_dense, (4.0, CArray([4, 0, 6]), 4.0))

        _check_median(self.row_dense, (4.0, CArray([[4, 0, 6]]), 4.0))

        _check_median(self.column_dense, (4.0, 4.0, CArray([[4], [0], [6]])))

        _check_median(self.single_flat_dense, (4, 4, 4))
        _check_median(self.single_dense, (4, 4, 4))

        with self.assertRaises(NotImplementedError):
            self.array_sparse.median()

    def test_logical_and(self):
        """Test for CArray.logical_and() method."""
        self.logger.info("Test for CArray.logical_and() method.")

        def _logical_and(array1, array2, expected):

            self.logger.info("a1: \n{:}".format(array1))
            self.logger.info("a2: \n{:}".format(array2))

            logical_and_res = array1.logical_and(array2)
            self.logger.info("a1.logical_and(a2): \n{:}".format(logical_and_res))

            self.assertTrue((logical_and_res == expected).all())

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
        _logical_and(self.array_sparse_bool_false, self.array_sparse_bool_false,
                     self.array_sparse_bool_false.astype(bool))
        _logical_and(self.array_dense_bool_false, self.array_dense_bool_false,
                     self.array_dense_bool_false.astype(bool))

        _logical_and(self.empty_sparse, self.empty_sparse, self.empty_sparse)
        _logical_and(self.empty_flat_dense, self.empty_flat_dense, self.empty_flat_dense)
        
    def test_logical_or(self):
        """Test for CArray.logical_or() method."""
        self.logger.info("Test for CArray.logical_or() method.")
        
        def _logical_or(array1, array2, expected):

            self.logger.info("a1: \n{:}".format(array1))
            self.logger.info("a2: \n{:}".format(array2))

            logical_or_res = array1.logical_or(array2)
            self.logger.info("a1.logical_or(a2): \n{:}".format(logical_or_res))

            self.assertTrue((logical_or_res == expected).all())

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
        _logical_or(self.empty_flat_dense, self.empty_flat_dense, self.empty_flat_dense)

    def test_logical_not(self):
        """Test for CArray.logical_not() method."""
        self.logger.info("Test for CArray.logical_not() method.")

        def _logical_not(array, expected):

            self.logger.info("a: \n{:}".format(array))

            logical_not_res = array.logical_not()
            self.logger.info("a.logical_not(): \n{:}".format(logical_not_res))

            self.assertTrue((logical_not_res == expected).all())

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

    def test_dot(self):
        """"Test for CArray.dot() method."""
        self.logger.info("Test for CArray.dot() method.")
        s_vs_s = self.array_sparse.dot(self.array_sparse.T)
        s_vs_d = self.array_sparse.dot(self.array_dense.T)
        d_vs_d = self.array_dense.dot(self.array_dense.T)
        d_vs_s = self.array_dense.dot(self.array_sparse.T)

        # Check if method returned correct datatypes
        self.assertIsInstance(s_vs_s._data, Csparse)
        self.assertIsInstance(s_vs_d._data, Csparse)
        self.assertIsInstance(d_vs_d._data, Cdense)
        self.assertIsInstance(d_vs_s._data, Cdense)

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

    def test_operators_array_vs_array(self):
        """Test for mathematical operators array vs array."""
        operators = [op.add, op.sub]
        expected_result = [Csparse, Cdense, Cdense, Cdense]
        items = [(self.array_sparse, self.array_sparse),
                 (self.array_sparse, self.array_dense),
                 (self.array_dense, self.array_sparse),
                 (self.array_dense, self.array_dense)]
        self._test_cycle(operators, items, expected_result)

        operators = [op.mul]
        expected_result = [Csparse, Csparse, Cdense, Cdense]
        items = [(self.array_sparse, self.array_sparse),
                 (self.array_sparse, self.array_dense),
                 (self.array_dense, self.array_sparse),
                 (self.array_dense, self.array_dense)]
        self._test_cycle(operators, items, expected_result)

        operators = [op.div]
        expected_result = [Cdense, Cdense, Cdense, Cdense]
        items = [(self.array_sparse, self.array_sparse),
                 (self.array_sparse, self.array_dense),
                 (self.array_dense, self.array_sparse),
                 (self.array_dense, self.array_dense)]
        self._test_cycle(operators, items, expected_result)

        operators = [op.pow, CArray.pow]
        expected_result = [Cdense, Cdense]
        items = [(self.array_dense, self.array_sparse),
                 (self.array_dense, self.array_dense)]
        self._test_cycle(operators, items, expected_result)

        # Sparse array ** array is not supported
        with self.assertRaises(TypeError):
            self.array_sparse ** self.array_sparse
        with self.assertRaises(TypeError):
            self.array_sparse ** self.array_dense
        with self.assertRaises(TypeError):
            self.array_sparse.pow(self.array_sparse)
        with self.assertRaises(TypeError):
            self.array_sparse.pow(self.array_dense)

    def test_operators_array(self):
        """Test for mathematical operators for single array."""
        # abs()
        self.logger.info("Checking abs() operator...")
        s_abs = abs(self.array_sparse)
        d_abs = abs(self.array_dense)
        # Check if method returned correct datatypes
        self.assertIsInstance(s_abs._data, Csparse)
        self.assertIsInstance(d_abs._data, Cdense)
        # Check if we have the same output in all cases
        self.assertTrue(self._test_multiple_eq([s_abs, d_abs]))

        # array.abs()
        self.logger.info("Checking .abs() method...")
        s_abs = self.array_sparse.abs()
        d_abs = self.array_dense.abs()
        # Check if method returned correct datatypes
        self.assertIsInstance(s_abs._data, Csparse)
        self.assertIsInstance(d_abs._data, Cdense)
        # Check if we have the same output in all cases
        self.assertTrue(self._test_multiple_eq([s_abs, d_abs]))

        # Negative
        self.logger.info("Checking negative operator...")
        s_abs = -self.array_sparse
        d_abs = -self.array_dense
        # Check if method returned correct datatypes
        self.assertIsInstance(s_abs._data, Csparse)
        self.assertIsInstance(d_abs._data, Cdense)
        # Check if we have the same output in all cases
        self.assertTrue(self._test_multiple_eq([s_abs, d_abs]))

    def test_operators_array_vs_scalar(self):
        """Test for mathematical operators array vs scalar."""

        # ARRAY +,* SCALAR, SCALAR +,* ARRAY
        operators = [op.add, op.mul]
        expected_result = [Cdense, Cdense,
                           Cdense, Cdense,
                           Cdense, Cdense,
                           Cdense, Cdense]
        items = [(self.array_dense, 2), (2, self.array_dense),
                 (self.array_dense, np.ravel(2)[0]),
                 (np.ravel(2)[0], self.array_dense),
                 (self.array_dense, np.ravel(2.0)[0]),
                 (np.ravel(2.0)[0], self.array_dense),
                 (self.array_dense, np.float32(2.0)),
                 (np.float32(2.0), self.array_dense)]
        self._test_cycle(operators, items, expected_result)

        # ARRAY * SCALAR, SCALAR * ARRAY
        operators = [op.mul]
        expected_result = [Csparse, Csparse,
                           Csparse, Csparse,
                           Csparse, Csparse,
                           Csparse, Csparse]
        items = [(self.array_sparse, 2),
                 (2, self.array_sparse),
                 (self.array_sparse, np.ravel(2)[0]),
                 (np.ravel(2)[0], self.array_sparse),
                 (self.array_sparse, np.ravel(2.0)[0]),
                 (np.ravel(2.0)[0], self.array_sparse),
                 (self.array_sparse, np.float32(2.0)),
                 (np.float32(2.0), self.array_sparse)]
        self._test_cycle(operators, items, expected_result)

        # ARRAY / SCALAR
        operators = [op.div]
        expected_result = [Csparse, Csparse, Csparse, Csparse]
        items = [(self.array_sparse, 2),
                 (self.array_sparse, np.ravel(2)[0]),
                 (self.array_sparse, np.ravel(2.0)[0]),
                 (self.array_sparse, np.float32(2))]
        self._test_cycle(operators, items, expected_result)

        # ARRAY -,/ SCALAR
        operators = [op.sub, op.div]
        expected_result = [Cdense, Cdense, Cdense, Cdense]
        items = [(self.array_dense, 2),
                 (self.array_dense, np.ravel(2)[0]),
                 (self.array_dense, np.ravel(2.0)[0]),
                 (self.array_dense, np.float32(2))]
        self._test_cycle(operators, items, expected_result)

        # SCALAR -,/ ARRAY
        operators = [op.sub, op.div]
        expected_result = [Cdense, Cdense, Cdense, Cdense]
        items = [(2, self.array_dense),
                 (np.ravel(2)[0], self.array_dense),
                 (np.ravel(2.0)[0], self.array_dense),
                 (np.float32(2), self.array_dense)]
        self._test_cycle(operators, items, expected_result)

        # ARRAY ** SCALAR
        operators = [op.pow, CArray.pow]
        expected_result = [Csparse, Cdense,
                           Csparse, Cdense,
                           Csparse, Cdense,
                           Csparse, Cdense]
        items = [(self.array_sparse, 2), (self.array_dense, 2),
                 (self.array_sparse, np.ravel(2)[0]),
                 (self.array_dense, np.ravel(2)[0]),
                 (self.array_sparse, np.ravel(2.0)[0]),
                 (self.array_dense, np.ravel(2.0)[0]),
                 (self.array_sparse, np.float32(2)),
                 (self.array_dense, np.float32(2))]
        self._test_cycle(operators, items, expected_result)

        # SCALAR ** ARRAY
        operators = [op.pow]
        expected_result = [Cdense, Cdense, Cdense, Cdense]
        items = [(2, self.array_dense),
                 (np.ravel(2)[0], self.array_dense),
                 (np.ravel(2.0)[0], self.array_dense),
                 (np.float32(2), self.array_dense)]
        self._test_cycle(operators, items, expected_result)

        # SCALAR / SPARSE ARRAY NOT SUPPORTED
        with self.assertRaises(NotImplementedError):
            2 / self.array_sparse
        with self.assertRaises(NotImplementedError):
            np.ravel(2)[0] / self.array_sparse
        with self.assertRaises(NotImplementedError):
            np.ravel(2.0)[0] / self.array_sparse
        with self.assertRaises(NotImplementedError):
            np.float32(2) / self.array_sparse

        # SCALAR ** SPARSE ARRAY NOT SUPPORTED
        with self.assertRaises(NotImplementedError):
            2 ** self.array_sparse
        with self.assertRaises(NotImplementedError):
            np.ravel(2)[0] ** self.array_sparse
        with self.assertRaises(NotImplementedError):
            np.ravel(2.0)[0] ** self.array_sparse
        with self.assertRaises(NotImplementedError):
            np.float32(2) ** self.array_sparse

    def test_operators_array_vs_unsupported(self):
        """Test for mathematical operators array vs unsupported types."""

        def test_unsupported(x):
            for operator in [op.add, op.sub, op.mul, op.div, op.pow]:
                with self.assertRaises(TypeError):
                    self.logger.info("Testing {:} dense vs '{:}'".format(
                        operator.__name__, type(x).__name__))
                    operator(self.array_dense, x)
                with self.assertRaises(TypeError):
                    self.logger.info("Testing {:} sparse vs '{:}'".format(
                        operator.__name__, type(x).__name__))
                    operator(self.array_sparse, x)
                with self.assertRaises(TypeError):
                    self.logger.info("Testing {:} dense vect vs '{:}'".format(
                        operator.__name__, type(x).__name__))
                    operator(self.row_flat_dense, x)
                with self.assertRaises(TypeError):
                    self.logger.info("Testing {:} sparse vect vs '{:}'".format(
                        operator.__name__, type(x).__name__))
                    operator(self.row_sparse, x)

        test_unsupported(np.array([1, 2, 3]))
        test_unsupported(scs.csr_matrix([1, 2, 3]))
        test_unsupported([1, 2, 3])
        test_unsupported((1, 2, 3))
        test_unsupported(set([1, 2, 3]))
        test_unsupported(dict({1: 2}))
        test_unsupported('test')

    def test_operators_unsupported_vs_array(self):
        """Test for mathematical operators unsupported types vs array."""

        def test_unsupported(x):
            for operator in [op.add, op.sub, op.mul, op.div, op.pow]:
                with self.assertRaises(TypeError):
                    self.logger.info("Testing {:} '{:}' vs dense".format(
                        operator.__name__, type(x).__name__))
                    operator(x, self.array_dense)
                with self.assertRaises(TypeError):
                    self.logger.info("Testing {:} '{:}' vs sparse".format(
                        operator.__name__, type(x).__name__))
                    operator(x, self.array_sparse)
                with self.assertRaises(TypeError):
                    self.logger.info("Testing {:} '{:}' vs dense vect".format(
                        operator.__name__, type(x).__name__))
                    operator(x, self.row_flat_dense)
                with self.assertRaises(TypeError):
                    self.logger.info("Testing {:} '{:}' vs sparse vect".format(
                        operator.__name__, type(x).__name__))
                    operator(x, self.row_sparse)

        # Array do broadcasting of each element wrt our array
        # There is NO way of blocking this
        # test_unsupported(np.array([1, 2, 3]))
        # test_unsupported(scs.csr_matrix([1, 2, 3]))

        test_unsupported([1, 2, 3])
        test_unsupported((1, 2, 3))
        test_unsupported(set([1, 2, 3]))
        test_unsupported(dict({1: 2}))
        test_unsupported('test')

    def test_comparison_array_vs_array(self):
        """Test for comparison operators array vs array."""
        operators = [op.eq, op.lt, op.le, op.gt, op.ge, op.ne]
        expected_result = [Csparse, Cdense, Cdense, Cdense]
        items = [(self.array_sparse, self.array_sparse),
                 (self.array_sparse, self.array_dense),
                 (self.array_dense, self.array_sparse),
                 (self.array_dense, self.array_dense)]
        self._test_cycle(operators, items, expected_result)

    def test_comparison_array_vs_scalar(self):
        """Test for comparison operators array vs scalar."""
        operators = [op.eq, op.lt, op.le, op.gt, op.ge, op.ne]
        expected_result = [Csparse, Cdense, Csparse, Cdense]
        items = [(self.array_sparse, 2),
                 (self.array_dense, 2),
                 (self.array_sparse, np.ravel(2)[0]),
                 (self.array_dense, np.ravel(2)[0])]
        self._test_cycle(operators, items, expected_result)

    def test_comparison_array_vs_unsupported(self):
        """Test for comparison operators array vs unsupported types."""

        def test_unsupported_arrays(x):
            for operator in [op.eq, op.lt, op.le, op.gt, op.ge, op.ne]:

                with self.assertRaises(TypeError):
                    self.logger.info("Testing {:} '{:}' vs dense".format(
                        operator.__name__, type(x).__name__))
                    operator(self.array_dense, x)

                with self.assertRaises(TypeError):
                    self.logger.info("Testing {:} '{:}' vs sparse".format(
                        operator.__name__, type(x).__name__))
                    operator(self.array_sparse, x)

                with self.assertRaises(TypeError):
                    self.logger.info("Testing {:} '{:}' vs dense vect".format(
                        operator.__name__, type(x).__name__))
                    operator(self.row_flat_dense, x)

                with self.assertRaises(TypeError):
                    self.logger.info("Testing {:} '{:}' vs sparse vect".format(
                        operator.__name__, type(x).__name__))
                    operator(self.row_sparse, x)

        def test_unsupported(x):
            for operator in [op.lt, op.le, op.gt, op.ge]:

                with self.assertRaises(TypeError):
                    self.logger.info("Testing {:} '{:}' vs dense".format(
                        operator.__name__, type(x).__name__))
                    operator(self.array_dense, x)

                with self.assertRaises(TypeError):
                    self.logger.info("Testing {:} '{:}' vs sparse".format(
                        operator.__name__, type(x).__name__))
                    operator(self.array_sparse, x)

                with self.assertRaises(TypeError):
                    self.logger.info("Testing {:} '{:}' vs dense vect".format(
                        operator.__name__, type(x).__name__))
                    operator(self.row_flat_dense, x)

                with self.assertRaises(TypeError):
                    self.logger.info("Testing {:} '{:}' vs sparse vect".format(
                        operator.__name__, type(x).__name__))
                    operator(self.row_sparse, x)

        def test_false(x):

            self.logger.info("Testing {:} dense vs '{:}'".format(
                op.eq.__name__, type(x).__name__))
            self.assertFalse(op.eq(self.array_dense, x))

            self.logger.info("Testing {:} sparse vs '{:}'".format(
                op.eq.__name__, type(x).__name__))
            self.assertFalse(op.eq(self.array_sparse, x))

            self.logger.info("Testing {:} dense vect vs '{:}'".format(
                op.eq.__name__, type(x).__name__))
            self.assertFalse(op.eq(self.row_flat_dense, x))

            self.logger.info("Testing {:} sparse vect vs '{:}'".format(
                op.eq.__name__, type(x).__name__))
            self.assertFalse(op.eq(self.row_sparse, x))

        def test_true(x):

            self.logger.info("Testing {:} dense vs '{:}'".format(
                op.ne.__name__, type(x).__name__))
            self.assertTrue(op.ne(self.array_dense, x))

            self.logger.info("Testing {:} sparse vs '{:}'".format(
                op.ne.__name__, type(x).__name__))
            self.assertTrue(op.ne(self.array_sparse, x))

            self.logger.info("Testing {:} dense vect vs '{:}'".format(
                op.ne.__name__, type(x).__name__))
            self.assertTrue(op.ne(self.row_flat_dense, x))

            self.logger.info("Testing {:} sparse vect vs '{:}'".format(
                op.ne.__name__, type(x).__name__))
            self.assertTrue(op.ne(self.row_sparse, x))

        test_unsupported_arrays(np.array([1, 2, 3]))
        test_unsupported_arrays(scs.csr_matrix([1, 2, 3]))

        test_unsupported([1, 2, 3])
        test_unsupported((1, 2, 3))
        test_unsupported(set([1, 2, 3]))
        test_unsupported(dict({1: 2}))
        test_unsupported('test')

        test_false([1, 2, 3])
        test_false((1, 2, 3))
        test_false(set([1, 2, 3]))
        test_false(dict({1: 2}))
        test_false('test')

        test_true([1, 2, 3])
        test_true((1, 2, 3))
        test_true(set([1, 2, 3]))
        test_true(dict({1: 2}))
        test_true('test')

    def test_operators_comparison_vs_array(self):
        """Test for comparison operators unsupported types vs array."""

        def test_unsupported(x):
            for operator in [op.lt, op.le, op.gt, op.ge]:

                with self.assertRaises(TypeError):
                    self.logger.info("Testing {:} '{:}' vs dense".format(
                        operator.__name__, type(x).__name__))
                    operator(x, self.array_dense)

                with self.assertRaises(TypeError):
                    self.logger.info("Testing {:} '{:}' vs sparse".format(
                        operator.__name__, type(x).__name__))
                    operator(x, self.array_sparse)

                with self.assertRaises(TypeError):
                    self.logger.info("Testing {:} '{:}' vs dense vect".format(
                        operator.__name__, type(x).__name__))
                    operator(x, self.row_flat_dense)

                with self.assertRaises(TypeError):
                    self.logger.info("Testing {:} '{:}' vs sparse vect".format(
                        operator.__name__, type(x).__name__))
                    operator(x, self.row_sparse)

        def test_false(x):

            self.logger.info("Testing {:} dense vs '{:}'".format(
                op.eq.__name__, type(x).__name__))
            self.assertFalse(op.eq(x, self.array_dense))

            self.logger.info("Testing {:} sparse vs '{:}'".format(
                op.eq.__name__, type(x).__name__))
            self.assertFalse(op.eq(x, self.array_sparse))

            self.logger.info("Testing {:} dense vect vs '{:}'".format(
                op.eq.__name__, type(x).__name__))
            self.assertFalse(op.eq(x, self.row_flat_dense))

            self.logger.info("Testing {:} sparse vect vs '{:}'".format(
                op.eq.__name__, type(x).__name__))
            self.assertFalse(op.eq(x, self.row_sparse))

        def test_true(x):

            self.logger.info("Testing {:} dense vs '{:}'".format(
                op.ne.__name__, type(x).__name__))
            self.assertTrue(op.ne(x, self.array_dense))

            self.logger.info("Testing {:} sparse vs '{:}'".format(
                op.ne.__name__, type(x).__name__))
            self.assertTrue(op.ne(x, self.array_sparse))

            self.logger.info("Testing {:} dense vect vs '{:}'".format(
                op.ne.__name__, type(x).__name__))
            self.assertTrue(op.ne(x, self.row_flat_dense))

            self.logger.info("Testing {:} sparse vect vs '{:}'".format(
                op.ne.__name__, type(x).__name__))
            self.assertTrue(op.ne(x, self.row_sparse))

        # Array do broadcasting of each element wrt our array
        # There is NO way of blocking this
        # test_unsupported(np.array([1, 2, 3]))
        # test_unsupported(scs.csr_matrix([1, 2, 3]))

        test_unsupported([1, 2, 3])
        test_unsupported((1, 2, 3))
        test_unsupported(set([1, 2, 3]))
        test_unsupported(dict({1: 2}))
        test_unsupported('test')

        test_false([1, 2, 3])
        test_false((1, 2, 3))
        test_false(set([1, 2, 3]))
        test_false(dict({1: 2}))
        test_false('test')

        test_true([1, 2, 3])
        test_true((1, 2, 3))
        test_true(set([1, 2, 3]))
        test_true(dict({1: 2}))
        test_true('test')

    def test_save_load(self):
        """Test save/load of CArray"""
        self.logger.info("UNITTEST - CArray - save/load")

        test_file = fm.join(fm.abspath(__file__), 'test.txt')
        test_file_2 = fm.join(fm.abspath(__file__), 'test2.txt')

        # Cleaning test files
        try:
            fm.remove_file(test_file)
            fm.remove_file(test_file_2)
        except (OSError, IOError) as e:
            self.logger.info(e.message)

        self.logger.info(
            "UNITTEST - CArray - Testing save/load for sparse matrix")

        self.array_sparse.save(test_file)

        # Saving to a file handle is not supported for sparse arrays
        with self.assertRaises(NotImplementedError):
            with open(test_file_2, 'w') as f:
                self.array_sparse.save(f)

        loaded_array_sparse = CArray.load(
            test_file, arrayformat='sparse', dtype=int)

        self.assertFalse((loaded_array_sparse != self.array_sparse).any(),
                         "Saved and loaded arrays (sparse) are not equal!")

        self.logger.info(
            "UNITTEST - Csparse - Testing save/load for dense matrix")

        self.array_dense.save(test_file, overwrite=True)

        loaded_array_dense = CArray.load(test_file, arrayformat='dense', dtype=int)

        self.assertFalse((loaded_array_dense != self.array_dense).any(),
                         "Saved and loaded arrays (sparse) are not equal!")

        # Checking sparse/dense equality between loaded data
        self.assertFalse((loaded_array_sparse.todense() != loaded_array_dense).any(),
                         "Loaded arrays are not equal!")

        # Only 'dense' and 'sparse' arrayformat are supported
        with self.assertRaises(ValueError):
            CArray.load(test_file, arrayformat='test')

        # Cleaning test files
        try:
            fm.remove_file(test_file)
            fm.remove_file(test_file_2)
        except (OSError, IOError) as e:
            self.logger.info(e.message)

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

                    res_expected = expected[res_idx]
                    if not isinstance(res_expected, CArray):
                        self.assertNotIsInstance(res, CArray)
                        res = CArray(res).round(2)[0]
                        self.assertEqual(res, res_expected)
                    else:
                        self.assertIsInstance(res, CArray)
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
                          (0, CArray([[1, 0, 0, 0]], tosparse=True),
                           CArray([[0], [0], [0]], tosparse=True)))
        _check_minmaxmean('min', self.array_dense,
                          (0, CArray([[1, 0, 0, 0]]), CArray([[0], [0], [0]])))

        _check_minmaxmean('min', self.row_flat_dense,
                          (0, CArray([4, 0, 6]), 0))

        _check_minmaxmean('min', self.row_sparse,
                          (0, CArray([[4, 0, 6]], tosparse=True), 0))
        _check_minmaxmean('min', self.row_dense,
                          (0, CArray([[4, 0, 6]]), 0))

        _check_minmaxmean('min', self.column_sparse,
                          (0, 0, CArray([[4], [0], [6]], tosparse=True)))
        _check_minmaxmean('min', self.column_dense,
                          (0, 0, CArray([[4], [0], [6]])))

        _check_minmaxmean('min', self.single_flat_dense, (4, 4, 4))
        _check_minmaxmean('min', self.single_dense, (4, 4, 4))
        _check_minmaxmean('min', self.single_sparse, (4, 4, 4))

        self.logger.info("Testing CArray.max()")

        _check_minmaxmean('max', self.array_sparse,
                          (6, CArray([[3, 6, 0, 5]], tosparse=True),
                           CArray([[5], [4], [6]], tosparse=True)))
        _check_minmaxmean('max', self.array_dense,
                          (6, CArray([[3, 6, 0, 5]]), CArray([[5], [4], [6]])))

        _check_minmaxmean('max', self.row_flat_dense,
                          (6, CArray([4, 0, 6]), 6))

        _check_minmaxmean('max', self.row_sparse,
                          (6, CArray([[4, 0, 6]], tosparse=True), 6))
        _check_minmaxmean('max', self.row_dense,
                          (6, CArray([[4, 0, 6]]), 6))

        _check_minmaxmean('max', self.column_sparse,
                          (6, 6, CArray([[4], [0], [6]], tosparse=True)))
        _check_minmaxmean('max', self.column_dense,
                          (6, 6, CArray([[4], [0], [6]])))

        _check_minmaxmean('max', self.single_flat_dense, (4, 4, 4))
        _check_minmaxmean('max', self.single_dense, (4, 4, 4))
        _check_minmaxmean('max', self.single_sparse, (4, 4, 4))

        self.logger.info("Testing CArray.mean()")

        _check_minmaxmean('mean', self.array_sparse,
                          (1.75, CArray([[2, 3.33, 0, 1.67]]),
                           CArray([[1.5], [1.5], [2.25]])))
        _check_minmaxmean('mean', self.array_dense,
                          (1.75, CArray([[2, 3.33, 0, 1.67]]),
                           CArray([[1.5], [1.5], [2.25]])))

        _check_minmaxmean('mean', self.row_flat_dense,
                          (3.33, CArray([4, 0, 6]), 3.33))

        _check_minmaxmean('mean', self.row_sparse,
                          (3.33, CArray([[4, 0, 6]]), 3.33))
        _check_minmaxmean('mean', self.row_dense,
                          (3.33, CArray([[4, 0, 6]]), 3.33))

        _check_minmaxmean('mean', self.column_sparse,
                          (3.33, 3.33, CArray([[4], [0], [6]])))
        _check_minmaxmean('mean', self.column_dense,
                          (3.33, 3.33, CArray([[4], [0], [6]])))

        _check_minmaxmean('mean', self.single_flat_dense, (4, 4, 4))
        _check_minmaxmean('mean', self.single_dense, (4, 4, 4))
        _check_minmaxmean('mean', self.single_sparse, (4, 4, 4))

    def test_nanmin_nanmax(self):
        """Test for CArray.nanmin(), CArray.nanmax() method."""
        self.logger.info(
            "Test for CArray.nanmin(), CArray.nanmax() method.")

        def _check_nanminnanmax(func, array, expected):
            # Adding few nans to array
            array = array.astype(float)  # Arrays with nans have float dtype
            array[0, 0] = np.nan

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

                    res_expected = expected[res_idx]
                    if not isinstance(res_expected, CArray):
                        self.assertNotIsInstance(res, CArray)
                        res = CArray(res).round(2)[0]
                    else:
                        self.assertIsInstance(res, CArray)
                        self.assertEqual(res.isdense, res_expected.isdense)
                        self.assertEqual(res.issparse,
                                         res_expected.issparse)
                        res = res.round(2)
                        if keepdims is False:
                            res_expected = res_expected.ravel()
                        self.assertEqual(res.shape, res_expected.shape)
                    # use numpy.testing to proper compare arrays with nans
                    np.testing.assert_equal(CArray(res).tondarray(),
                                            CArray(res_expected).tondarray())

        # array_dense = CArray([[1, 0, 0, 5], [2, 4, 0, 0], [3, 6, 0, 0]]
        # row_flat_dense = CArray([4, 0, 6])

        self.logger.info("Testing CArray.nanmin()")

        _check_nanminnanmax('nanmin', self.array_dense,
                            (0, CArray([[2, 0, 0, 0]]),
                             CArray([[0], [0], [0]])))

        _check_nanminnanmax('nanmin', self.row_flat_dense,
                            (0, CArray([np.nan, 0, 6]), 0))

        _check_nanminnanmax('nanmin', self.row_dense,
                            (0, CArray([[np.nan, 0, 6]]), 0))

        _check_nanminnanmax('nanmin', self.column_dense,
                            (0, 0, CArray([[np.nan], [0], [6]])))

        _check_nanminnanmax('nanmin', self.single_flat_dense,
                            (np.nan, np.nan, np.nan))
        _check_nanminnanmax('nanmin', self.single_dense,
                            (np.nan, np.nan, np.nan))

        self.logger.info("Testing CArray.nanmax()")

        _check_nanminnanmax('nanmax', self.array_dense,
                            (6, CArray([[3, 6, 0, 5]]),
                             CArray([[5], [4], [6]])))

        _check_nanminnanmax('nanmax', self.row_flat_dense,
                            (6, CArray([np.nan, 0, 6]), 6))

        _check_nanminnanmax('nanmax', self.row_dense,
                            (6, CArray([[np.nan, 0, 6]]), 6))

        _check_nanminnanmax('nanmax', self.column_dense,
                            (6, 6, CArray([[np.nan], [0], [6]])))

        _check_nanminnanmax('nanmax', self.single_flat_dense,
                            (np.nan, np.nan, np.nan))
        _check_nanminnanmax('nanmax', self.single_dense,
                            (np.nan, np.nan, np.nan))

        with self.assertRaises(NotImplementedError):
            self.array_sparse.nanmin()
        with self.assertRaises(NotImplementedError):
            self.array_sparse.nanmax()

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

                    res_expected = expected[res_idx]
                    if not isinstance(res_expected, CArray):
                        self.assertNotIsInstance(res, CArray)
                        self.assertIsInstance(res, type(res_expected))
                        self.assertEqual(res, res_expected)
                    else:
                        self.assertIsInstance(res, CArray)
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

        _check_sum(self.row_flat_dense, (10, CArray([4, 0, 6]), 10))
        _check_sum(self.row_dense, (10, CArray([[4, 0, 6]]), 10))
        _check_sum(self.row_sparse, (10, CArray([[4, 0, 6]]), 10))

        _check_sum(self.column_dense, (10, 10, CArray([[4], [0], [6]])))
        _check_sum(self.column_sparse, (10, 10, CArray([[4], [0], [6]])))

        _check_sum(self.single_flat_dense, (4, 4, 4))
        _check_sum(self.single_dense, (4, 4, 4))
        _check_sum(self.single_sparse, (4, 4, 4))

        _check_sum(self.single_bool_flat_dense, (1, 1, 1))
        _check_sum(self.single_bool_dense, (1, 1, 1))
        _check_sum(self.single_bool_sparse, (1, 1, 1))

        _check_sum(self.empty_flat_dense, (0.0, 0.0, 0.0))
        _check_sum(self.empty_dense, (0.0, 0.0, 0.0))
        _check_sum(self.empty_sparse, (0.0, 0.0, 0.0))

    def test_cusum(self):
        """Test for CArray.cumsum() method."""
        self.logger.info("Test for CArray.cumsum() method.")

        def _check_cumsum(array, expected):
            self.logger.info("Array:\n{:}".format(array))

            for res_idx, axis in enumerate([None, 0, 1]):

                res = array.cumsum(axis=axis)
                self.logger.info("array.cumsum(axis={:}):\n{:}".format(axis, res))

                res_expected = expected[res_idx]
                if not isinstance(res_expected, CArray):
                    self.assertNotIsInstance(res, CArray)
                else:
                    self.assertIsInstance(res, CArray)
                    self.assertEqual(res.isdense, res_expected.isdense)
                    self.assertEqual(res.issparse, res_expected.issparse)
                    self.assertEqual(res.shape, res_expected.shape)
                self.assertFalse((res != res_expected).any())

        # array_dense = CArray([[1, 0, 0, 5], [2, 4, 0, 0], [3, 6, 0, 0]])
        # row_flat_dense = CArray([4, 0, 6])

        self.logger.info("Testing CArray.cumsum()")

        _check_cumsum(self.array_dense,
                      (CArray([1, 1, 1, 6, 8, 12, 12, 12, 15, 21, 21, 21]),
                       CArray([[1, 0, 0, 5], [3, 4, 0, 5], [6, 10, 0, 5]]),
                       CArray([[1, 1, 1, 6], [2, 6, 6, 6], [3, 9, 9, 9]])))

        _check_cumsum(self.array_dense_bool,
                      (CArray([1, 1, 2, 3, 3, 3, 3, 3, 4, 5, 6, 7]),
                       CArray([[1, 0, 1, 1], [1, 0, 1, 1], [2, 1, 2, 2]]),
                       CArray([[1, 1, 2, 3], [0, 0, 0, 0], [1, 2, 3, 4]])))

        _check_cumsum(self.row_flat_dense,
                      (CArray([4, 4, 10]),
                       CArray([4, 0, 6]),
                       CArray([4, 4, 10])))
        _check_cumsum(self.row_dense,
                      (CArray([4, 4, 10]),
                       CArray([[4, 0, 6]]),
                       CArray([[4, 4, 10]])))

        _check_cumsum(self.column_dense,
                      (CArray([4, 4, 10]),
                       CArray([[4], [4], [10]]),
                       CArray([[4], [0], [6]])))

        _check_cumsum(self.single_flat_dense,
                      (CArray([4]), CArray([4]), CArray([4])))
        _check_cumsum(self.single_dense,
                      (CArray([4]), CArray([[4]]), CArray([[4]])))

        _check_cumsum(self.single_bool_flat_dense,
                      (CArray([1]), CArray([1]), CArray([1])))
        _check_cumsum(self.single_bool_dense,
                      (CArray([1]), CArray([[1]]), CArray([[1]])))

        _check_cumsum(self.empty_flat_dense,
                      (CArray([]), CArray([]), CArray([])))
        _check_cumsum(self.empty_dense,
                      (CArray([]), CArray([[]]), CArray([[]])))

        with self.assertRaises(NotImplementedError):
            _check_cumsum(self.array_sparse, ())

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

                        res_expected = expected[res_idx]
                        if not isinstance(res_expected, CArray):
                            self.assertNotIsInstance(res, CArray)
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
                            self.assertIsInstance(res, CArray)
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

        _check_prod(self.row_flat_dense, (0, CArray([4, 0, 6]), 0))
        _check_prod(self.row_dense, (0, CArray([[4, 0, 6]]), 0))
        _check_prod(self.row_sparse,
                    (0, CArray([[4, 0, 6]], tosparse=True), 0))

        _check_prod(self.column_dense, (0, 0, CArray([[4], [0], [6]])))
        _check_prod(self.column_sparse,
                    (0, 0, CArray([[4], [0], [6]], tosparse=True)))

        _check_prod(self.single_flat_dense, (4, 4, 4))
        _check_prod(self.single_dense, (4, 4, 4))
        _check_prod(self.single_sparse, (4, 4, 4))

        _check_prod(self.single_bool_flat_dense, (1, 1, 1))
        _check_prod(self.single_bool_dense, (1, 1, 1))
        _check_prod(self.single_bool_sparse, (1, 1, 1))

        _check_prod(self.empty_flat_dense, (1.0, 1.0, 1.0))
        _check_prod(self.empty_dense, (1.0, 1.0, 1.0))
        _check_prod(self.empty_sparse, (1.0, 1.0, 1.0))

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

    def test_bincount(self):
        """Test for CArray.bincount() method."""
        self.logger.info("Test for CArray.bincount() method.")

        def _check_bincount(array, expected):
            self.logger.info("Array:\n{:}".format(array))

            res = array.bincount()
            self.logger.info("array.bincount():\n{:}".format(res))

            self.assertEqual(res.ndim, 1)
            self.assertEqual(res.size, array.max()+1)
            self.assertFalse((res != expected).any())

        with self.assertRaises(ValueError):
            self.row_dense.bincount()
        with self.assertRaises(ValueError):
            self.array_dense.bincount()
        with self.assertRaises(ValueError):
            # NotImplementedError is not raised as ValueError is raised first
            self.array_sparse.bincount()

        _check_bincount(self.row_flat_dense, CArray([1, 0, 0, 0, 1, 0, 1]))
        _check_bincount(self.single_flat_dense, CArray([0, 0, 0, 0, 1]))
        _check_bincount(self.single_bool_flat_dense, CArray([0, 1]))

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

    def test_find(self):
        """Test for CArray.find() method."""
        self.logger.info("Test for CArray.find() method.")

        self.logger.info("a: \n{:}".format(self.row_dense))
        greater_than_two = self.row_flat_dense.find(self.row_dense > 2)
        self.logger.info("a.find(a > 2): \n{:}".format(greater_than_two))
        self.assertTrue(greater_than_two == [0, 2])

        self.logger.info("a: \n{:}".format(self.row_sparse))
        greater_than_two = self.row_flat_dense.find(self.row_sparse > 2)
        self.logger.info("a.find(a > 2): \n{:}".format(greater_than_two))
        self.assertTrue(greater_than_two == [0, 2])

        self.logger.info("a: \n{:}".format(self.row_flat_dense))
        greater_than_two = self.row_flat_dense.find(self.row_flat_dense > 2)
        self.logger.info("a.find(a > 2): \n{:}".format(greater_than_two))
        self.assertTrue(greater_than_two == [0, 2])

        with self.assertRaises(ValueError):
            self.array_dense.find(self.array_dense > 2)

    def test_find_2d(self):
        """Test for CArray.find_2d() method."""
        self.logger.info("Test for CArray.test_find_2d() method.")

        self.logger.info("a: \n{:}".format(self.array_dense))
        greater_than_two = self.array_dense.find_2d(self.array_dense > 2)
        self.logger.info("a.find_2d(a > 2): \n{:}".format(greater_than_two))
        self.assertTrue(greater_than_two == [[0, 1, 2, 2], [3, 1, 0, 1]])

        self.logger.info("a: \n{:}".format(self.array_sparse))
        greater_than_two = self.array_dense.find_2d(self.array_sparse > 2)
        self.logger.info("a.find_2d(a > 2): \n{:}".format(greater_than_two))
        self.assertTrue(greater_than_two == [[0, 1, 2, 2], [3, 1, 0, 1]])

        self.logger.info("a: \n{:}".format(self.row_dense))
        greater_than_two = self.row_flat_dense.find_2d(self.row_dense > 2)
        self.logger.info("a.find_2d(a > 2): \n{:}".format(greater_than_two))
        self.assertTrue(greater_than_two == [[0, 0], [0, 2]])

        self.logger.info("a: \n{:}".format(self.row_sparse))
        greater_than_two = self.row_flat_dense.find_2d(self.row_sparse > 2)
        self.logger.info("a.find_2d(a > 2): \n{:}".format(greater_than_two))
        self.assertTrue(greater_than_two == [[0, 0], [0, 2]])

        self.logger.info("a: \n{:}".format(self.row_flat_dense))
        greater_than_two = self.row_flat_dense.find_2d(self.row_flat_dense > 2)
        self.logger.info("a.find_2d(a > 2): \n{:}".format(greater_than_two))
        self.assertTrue(greater_than_two == [[0, 0], [0, 2]])

        self.logger.info("a: \n{:}".format(self.array_dense))
        greater_than_nn = self.array_dense.find_2d(self.array_dense > 99)
        self.logger.info("a.find_2d(a > 99): \n{:}".format(greater_than_nn))
        self.assertTrue(greater_than_nn == [[], []])

        self.logger.info("a: \n{:}".format(self.array_sparse))
        greater_than_nn = self.array_dense.find_2d(self.array_sparse > 99)
        self.logger.info("a.find_2d(a > 99): \n{:}".format(greater_than_nn))
        self.assertTrue(greater_than_nn == [[], []])

        self.logger.info("a: \n{:}".format(self.empty_dense))
        greater_than_nn = self.empty_dense.find_2d(self.empty_dense > 99)
        self.logger.info("a.find_2d(a > 99): \n{:}".format(greater_than_nn))
        self.assertTrue(greater_than_nn == [[], []])

        self.logger.info("a: \n{:}".format(self.empty_sparse))
        greater_than_nn = self.empty_dense.find_2d(self.empty_sparse > 99)
        self.logger.info("a.find_2d(a > 99): \n{:}".format(greater_than_nn))
        self.assertTrue(greater_than_nn == [[], []])

    def test_binary_search(self):
        """Test for CArray.binary_search() method."""
        self.logger.info("Test for CArray.binary_search() method.")

        def _check_binary_search(a):
            self.logger.info("array: \n{:}".format(a))
            self.assertEqual(a.binary_search(-6), 0)  # Out of minimum
            self.assertEqual(a.binary_search(1), 0)  # Exact value
            self.assertEqual(a.binary_search(2.2), 1)  # Near value (after)
            self.assertEqual(a.binary_search(3.9), 3)  # Near value (before)
            self.assertEqual(a.binary_search(6), 3)  # Out of maximum

        _check_binary_search(CArray([1, 2.4, 3, 4.3]))
        _check_binary_search(CArray([[1, 2.4, 3, 4.3]]))
        _check_binary_search(CArray([[1, 2.4, 3, 4.3]], tosparse=True))
        _check_binary_search(CArray([[1], [2.4], [3], [4.3]]))
        _check_binary_search(CArray([[1], [2.4], [3], [4.3]], tosparse=True))
        _check_binary_search(CArray([[1, 2.4], [3, 4.3]]))
        _check_binary_search(CArray([[1, 2.4], [3, 4.3]], tosparse=True))

    def test_todense(self):
        """Test for CArray.todense() method."""
        self.logger.info("Test for CArray.todense() method")

        def _check_todense(array):
            self.logger.info("array:\n{:}".format(array))

            array_dense = array.todense()
            self.logger.info("array.todense():\n{:}".format(array_dense))

            self.assertTrue(array_dense.isdense)
            self.assertEqual(array_dense.size, array.size)
            self.assertEqual(array_dense.shape, array.shape)

        # Sparse arrays
        _check_todense(self.array_sparse)
        _check_todense(self.row_sparse)
        _check_todense(self.column_sparse)

        # Dense arrays
        _check_todense(self.array_dense)
        _check_todense(self.row_flat_dense)
        _check_todense(self.row_dense)
        _check_todense(self.column_dense)

        # Bool arrays
        _check_todense(self.array_dense_bool)
        _check_todense(self.array_sparse_bool)

        _check_todense(self.single_flat_dense)
        _check_todense(self.single_dense)
        _check_todense(self.single_sparse)

        _check_todense(self.empty_dense)
        _check_todense(self.empty_flat_dense)
        _check_todense(self.empty_sparse)

        with self.assertRaises(ValueError):
            self.array_dense.todense(dtype=int)
        with self.assertRaises(ValueError):
            self.array_dense.todense(shape=())
        with self.assertRaises(ValueError):
            self.array_dense.todense(dtype=int, shape=())

    def test_tosparse(self):
        """Test for CArray.tosparse() method."""
        self.logger.info("Test for CArray.tosparse() method")

        def _check_tosparse(array):
            self.logger.info("array:\n{:}".format(array))

            array_sparse = array.tosparse()
            self.logger.info("array.tosparse():\n{:}".format(array_sparse))

            self.assertTrue(array_sparse.issparse)
            self.assertEqual(array_sparse.size, array.size)
            self.assertEqual(array_sparse.is_vector_like, array.is_vector_like)

            if array.ndim > 1:  # original array is 2D, shape must not change
                self.assertEqual(array_sparse.shape, array.shape)

        # Sparse arrays
        _check_tosparse(self.array_sparse)
        _check_tosparse(self.row_sparse)
        _check_tosparse(self.column_sparse)

        # Dense arrays
        _check_tosparse(self.array_dense)
        _check_tosparse(self.row_flat_dense)
        _check_tosparse(self.row_dense)
        _check_tosparse(self.column_dense)

        # Bool arrays
        _check_tosparse(self.array_dense_bool)
        _check_tosparse(self.array_sparse_bool)

        _check_tosparse(self.single_flat_dense)
        _check_tosparse(self.single_dense)
        _check_tosparse(self.single_sparse)

        _check_tosparse(self.empty_dense)
        _check_tosparse(self.empty_flat_dense)
        _check_tosparse(self.empty_sparse)

        with self.assertRaises(ValueError):
            self.array_sparse.tosparse(dtype=int)
        with self.assertRaises(ValueError):
            self.array_sparse.tosparse(shape=())
        with self.assertRaises(ValueError):
            self.array_sparse.tosparse(dtype=int, shape=())

    def test_tondarray_tocsr(self):
        """Test for CArray.tondarray(), CArray.tocsr() method."""
        self.logger.info("Test for CArray.tondarray(), CArray.tocsr() method.")

        def _check_tondarray_tocsr(array):
            self.logger.info("array:\n{:}".format(array))

            ndarray = array.tondarray()
            self.logger.info("array.tondarray():\n{:}".format(ndarray))
            csr = array.tocsr()
            self.logger.info("array.tocsr():\n{:}".format(csr))

            self.assertIsInstance(ndarray, np.ndarray)
            self.assertIsInstance(csr, scs.csr_matrix)

            self.assertEqual(array.size, ndarray.size)
            self.assertEqual(array.size, csr.shape[0] * csr.shape[1])

            self.assertEqual(array.shape, ndarray.shape)
            if array.isdense:  # flat dense arrays become 2D when sparse
                self.assertEqual(array.atleast_2d().shape, csr.shape)

        # Sparse arrays
        _check_tondarray_tocsr(self.array_sparse)
        _check_tondarray_tocsr(self.row_sparse)
        _check_tondarray_tocsr(self.column_sparse)

        # Dense arrays
        _check_tondarray_tocsr(self.array_dense)
        _check_tondarray_tocsr(self.row_flat_dense)
        _check_tondarray_tocsr(self.row_dense)
        _check_tondarray_tocsr(self.column_dense)

        # Bool arrays
        _check_tondarray_tocsr(self.array_dense_bool)
        _check_tondarray_tocsr(self.array_sparse_bool)

        _check_tondarray_tocsr(self.single_flat_dense)
        _check_tondarray_tocsr(self.single_dense)
        _check_tondarray_tocsr(self.single_sparse)

        _check_tondarray_tocsr(self.empty_dense)
        _check_tondarray_tocsr(self.empty_flat_dense)
        _check_tondarray_tocsr(self.empty_sparse)
        
    def test_tolist(self):
        """Test for CArray.tolist() method."""
        self.logger.info("Test for CArray.tolist() method.")

        def _check_tolist(array):
            self.logger.info("array:\n{:}".format(array))

            array_list = array.tolist()
            self.logger.info("array.tolist():\n{:}".format(array_list))

            self.assertIsInstance(array_list, list)

            self.assertEqual(len(array_list), array.shape[0])
            if array.ndim > 1:
                for elem in array_list:
                    self.assertEqual(len(elem), array.shape[1])

        # Sparse arrays
        _check_tolist(self.array_sparse)
        _check_tolist(self.row_sparse)
        _check_tolist(self.column_sparse)

        # Dense arrays
        _check_tolist(self.array_dense)
        _check_tolist(self.row_flat_dense)
        _check_tolist(self.row_dense)
        _check_tolist(self.column_dense)

        # Bool arrays
        _check_tolist(self.array_dense_bool)
        _check_tolist(self.array_sparse_bool)

        _check_tolist(self.single_flat_dense)
        _check_tolist(self.single_dense)
        _check_tolist(self.single_sparse)

        _check_tolist(self.empty_dense)
        _check_tolist(self.empty_flat_dense)
        _check_tolist(self.empty_sparse)

    def test_non_zero_indices(self):
        """Property test non_zero_indices."""        
        self.logger.info("Testing non_zero_indices property")
        
        def non_zero_indices(self, structure_name, matrix, row_vector, column_vector):

            self.logger.info("nnz_indices: matrix \n" + str(matrix))        
            self.logger.info("Non zero index are: \n" + str(matrix.nnz_indices))
            self.assertEquals(matrix.nnz_indices == [[0, 0, 1, 1, 2, 2], [0, 3, 0, 1, 0, 1]],
                              True, "nnz_indices returned the wrong indices indices")
     
            self.assertEquals(isinstance(matrix.nnz_indices, list), True, "nnz_indices not returned a list")
            self.assertEquals(len(matrix.nnz_indices), 2, "nnz_indices not returned a list of 2 element")
            self.assertEquals(all(isinstance(elem, list) for elem in matrix.nnz_indices), True,
                              "nnz_indices not returned a list of 2 lists")
          
        non_zero_indices(self, "sparse", self.array_sparse, self.row_sparse, self.column_sparse)
        non_zero_indices(self, "dense", self.array_dense, self.row_sparse, self.column_dense)
 
    def test_round(self):
        """Test for CArray.round() method."""
        self.logger.info("Test for CArray.round() method.")
        
        def _round(array):
                                    
            array_float = array.astype(float)
            array_float *= 1.0201
            self.logger.info("a: \n{:}".format(array))

            round_res = array_float.round()
            self.logger.info("a.round(): \n{:}".format(round_res))
            self.assertTrue((round_res == array.astype(float)).all())

            round_res = array_float.round(decimals=2)
            self.logger.info("a.round(decimals=2): \n{:}".format(round_res))
            array_test = array * 1.02
            self.assertTrue((round_res == array_test).all())

            round_res = array_float.round(decimals=6)
            self.logger.info("a.round(decimals=6): \n{:}".format(round_res))
            self.assertTrue((round_res == array_float).all())

        _round(self.array_sparse)
        _round(self.row_sparse)
        _round(self.column_sparse)
        _round(self.array_dense)
        _round(self.row_sparse)
        _round(self.column_dense)

    def test_non_zero(self):
        """Test for CArray.nnz, CArray.nnz_data properties."""
        self.logger.info("Testing CArray.nnz property")

        def check_nnz(array):
            self.logger.info("array:\n{:}".format(array))
            self.logger.info("nnz: {:}".format(array.nnz))
            self.assertEquals(array.nnz, array.nnz_data.size)

        check_nnz(self.array_sparse)
        check_nnz(self.row_sparse)
        check_nnz(self.column_sparse)
        check_nnz(self.array_dense)
        check_nnz(self.row_dense)
        check_nnz(self.column_dense)

        def check_nnz_data(array, expected_nnz):
            self.logger.info("array:\n{:}".format(array))
            self.logger.info("nnz: {:}".format(array.nnz))
            self.assertFalse((array.nnz_data != expected_nnz).any())

        self.logger.info("Testing CArray.nnz_data property")
        check_nnz_data(self.array_sparse, CArray([1, 5, 2, 4, 3, 6]))
        check_nnz_data(self.row_sparse, CArray([4, 6]))
        check_nnz_data(self.column_sparse, CArray([4, 6]))
        check_nnz_data(self.array_dense, CArray([1, 5, 2, 4, 3, 6]))
        check_nnz_data(self.row_dense, CArray([4, 6]))
        check_nnz_data(self.column_dense, CArray([4, 6]))

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
                self.assertEquals(concat_res.ndim, 1)
            else:  # ... but if array is sparse let's check for shape[0]
                self.assertEquals(concat_res.shape[0], 1)
            self.assertTrue((concat_res[:array1.size] == array1.ravel()).all())
            self.assertTrue((concat_res[array1.size:] == array2.ravel()).all())

            array1_shape0 = array1.atleast_2d().shape[0]
            array1_shape1 = array1.atleast_2d().shape[1]
            array2_shape0 = array2.atleast_2d().shape[0]
            array2_shape1 = array2.atleast_2d().shape[1]

            # check append on axis 0 (vertical)
            concat_res = CArray.concatenate(array1, array2, axis=0)
            self.logger.info("concat(a1, a2, axis=0): {:}".format(concat_res))
            self.assertEquals(concat_res.shape[1], array1_shape1)
            self.assertEquals(
                concat_res.shape[0], array1_shape0 + array2_shape0)
            self.assertTrue((concat_res[:array1_shape0, :] == array1).all())
            self.assertTrue((concat_res[array1_shape0:, :] == array2).all())

            # check append on axis 1 (horizontal)
            concat_res = CArray.concatenate(array1, array2, axis=1)
            self.logger.info("concat(a1, a2, axis=1): {:}".format(concat_res))
            self.assertEquals(
                concat_res.shape[1], array1_shape1 + array2_shape1)
            self.assertEquals(concat_res.shape[0], array1_shape0)
            self.assertTrue((concat_res[:, :array1_shape1] == array1).all())
            self.assertTrue((concat_res[:, array1_shape1:] == array2).all())

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
                self.assertEquals(append_res.ndim, 1)
            else:  # ... but if array is sparse let's check for shape[0]
                self.assertEquals(append_res.shape[0], 1)
            self.assertTrue((append_res[:array1.size] == array1.ravel()).all())
            self.assertTrue((append_res[array1.size:] == array2.ravel()).all())

            array1_shape0 = array1.atleast_2d().shape[0]
            array1_shape1 = array1.atleast_2d().shape[1]
            array2_shape0 = array2.atleast_2d().shape[0]
            array2_shape1 = array2.atleast_2d().shape[1]

            # check append on axis 0 (vertical)
            append_res = array1.append(array2, axis=0)
            self.logger.info("a1.append(a2, axis=0): {:}".format(append_res))
            self.assertEquals(append_res.shape[1], array1_shape1)
            self.assertEquals(
                append_res.shape[0], array1_shape0 + array2_shape0)
            self.assertTrue((append_res[array1_shape0:, :] == array2).all())

            # check append on axis 1 (horizontal)
            append_res = array1.append(array2, axis=1)
            self.logger.info("a1.append(a2, axis=1): {:}".format(append_res))
            self.assertEquals(
                append_res.shape[1], array1_shape1 + array2_shape1)
            self.assertEquals(append_res.shape[0], array1_shape0)
            self.assertTrue((append_res[:, array1_shape1:] == array2).all())

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

    def test_bool_operators(self):

        a = CArray([1, 2, 3])
        b = CArray([1, 1, 1])

        d = (a < 2)
        c = (b == 1)

        self.logger.info("C -> " + str(c))
        self.logger.info("D -> " + str(d))

        self.logger.info("C logical_and D -> " + str(c.logical_and(d)))
        self.logger.info("D logical_and C -> " + str(d.logical_and(c)))

        with self.assertRaises(ValueError):
            print d and c
        with self.assertRaises(ValueError):
            print c and d
        with self.assertRaises(ValueError):
            print d or c
        with self.assertRaises(ValueError):
            print c or d

        a = CArray(True)
        b = CArray(False)

        self.assertTrue((a and b) == False)
        self.assertTrue((b and a) == False)
        self.assertTrue((a or b) == True)
        self.assertTrue((b or a) == True)

    def test_iteration(self):
        """Unittest for CArray __iter__."""
        self.logger.info("Unittest for CArray __iter__")

        res = []
        for elem_id, elem in enumerate(self.array_dense):
            res.append(elem)
            self.assertFalse(self.array_dense.ravel()[elem_id] != elem)
        # Check if all array elements have been returned
        self.assertEquals(len(res), self.array_dense.size)

        res = []
        for elem_id, elem in enumerate(self.array_sparse):
            res.append(elem)
            self.assertFalse(self.array_sparse.ravel()[elem_id] != elem)
        # Check if all array elements have been returned
        self.assertEquals(len(res), self.array_sparse.size)

        res = []
        for elem_id, elem in enumerate(self.row_flat_dense):
            res.append(elem)
            self.assertFalse(self.row_flat_dense[elem_id] != elem)
        # Check if all array elements have been returned
        self.assertEquals(len(res), self.row_flat_dense.size)

        res = []
        for elem_id, elem in enumerate(self.row_dense):
            res.append(elem)
            self.assertFalse(self.row_dense[elem_id] != elem)
        # Check if all array elements have been returned
        self.assertEquals(len(res), self.row_dense.size)

        res = []
        for elem_id, elem in enumerate(self.row_sparse):
            res.append(elem)
            self.assertFalse(self.row_sparse[elem_id] != elem)
        # Check if all array elements have been returned
        self.assertEquals(len(res), self.row_sparse.size)

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

    def test_clip(self):
        """Test for CArray.clip() method."""
        self.logger.info("Test for CArray.clip() method.")

        def _check_clip(array, expected):
            self.logger.info("Array:\n{:}".format(array))

            intervals = [(0, 2), (0, np.inf), (-np.inf, 0)]

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
                   CArray([[0., -np.inf, -np.inf, 1.6094],
                           [0.6931, 1.3863, -np.inf, -np.inf],
                           [1.0986, 1.7918, -np.inf, -np.inf]]))
        _check_log(self.row_flat_dense, CArray([1.3863, -np.inf, 1.7918]))
        _check_log(self.row_dense, CArray([[1.3863, -np.inf, 1.7918]]))
        _check_log(self.column_dense, CArray([[1.3863], [-np.inf], [1.7918]]))
        _check_log(self.single_flat_dense, CArray([1.3863]))
        _check_log(self.single_dense, CArray([[1.3863]]))

    def test_log10(self):
        """Test for CArray.log10() method."""
        self.logger.info("Test for CArray.log10() method.")

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
                     CArray([[0., -np.inf, -np.inf, 0.6990],
                             [0.3010, 0.6021, -np.inf, -np.inf],
                             [0.4771, 0.7782, -np.inf, -np.inf]]))
        _check_log10(self.row_flat_dense, CArray([0.6021, -np.inf, 0.7782]))
        _check_log10(self.row_dense, CArray([[0.6021, -np.inf, 0.7782]]))
        _check_log10(self.column_dense,
                     CArray([[0.6021], [-np.inf], [0.7782]]))
        _check_log10(self.single_flat_dense, CArray([0.6021]))
        _check_log10(self.single_dense, CArray([[0.6021]]))

    def test_sqrt(self):
        """Test for CArray.sqrt() method."""
        self.logger.info("Test for CArray.sqrt() method.")

        def _check_sqrt(array, expected):
            self.logger.info("Array:\n{:}".format(array))

            res = array.sqrt()
            self.logger.info("array.sqrt():\n{:}".format(res))

            self.assertIsInstance(res, CArray)
            self.assertEqual(res.isdense, expected.isdense)
            self.assertEqual(res.issparse, expected.issparse)
            self.assertEqual(res.shape, expected.shape)
            self.assertEqual(res.dtype, expected.dtype)
            np.testing.assert_almost_equal(
                res.tondarray(), expected.tondarray(), decimal=4)

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
        _check_sqrt(CArray([4., 0., -3.]), CArray([2., 0., np.nan]))
        _check_sqrt(self.row_dense, CArray([[2., 0., 2.4495]]))
        _check_sqrt(self.row_sparse, CArray([[2., 0., 2.4495]], tosparse=True))
        _check_sqrt(CArray([[4., 0., -3.]]), CArray([[2., 0., np.nan]]))
        _check_sqrt(CArray([4., 0., -3.], tosparse=True),
                    CArray([[2., 0., np.nan]], tosparse=True))
        _check_sqrt(self.column_dense, CArray([[2.], [0.], [2.4495]]))
        _check_sqrt(self.column_sparse,
                    CArray([[2.], [0.], [2.4495]], tosparse=True))
        _check_sqrt(self.single_flat_dense, CArray([2.]))
        _check_sqrt(self.single_dense, CArray([[2.]]))
        _check_sqrt(self.single_sparse, CArray([[2.]], tosparse=True))

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
                    self.assertFalse((res != 1).any())

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
                                    0, res[0, min(n_cols-1, k-1)])
                            elif k < 0:
                                self.assertEqual(
                                    0, res[min(n_rows-1, abs(k)-1), 0])
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
                                ((res != 0).logical_and(res != 1)).any())

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

                if not isinstance(inter, tuple):
                    self.assertFalse((res < 0).any())
                    self.assertFalse((res >= inter).any())
                else:
                    self.assertFalse((res < inter[0]).any())
                    self.assertFalse((res >= inter[1]).any())

    def test_norm(self):
        """Test for CArray.norm() method."""
        self.logger.info("Test for CArray.norm() method.")

        def _check_norm(array):
            self.logger.info("array:\n{:}".format(array))

            for ord_idx, ord in enumerate((None, 'fro', np.inf, -np.inf,
                                           0, 1, -1, 2, -2, 3, -3)):

                if ord == 'fro':  # Frobenius is a matrix norm
                    self.logger.info(
                        "array.norm(ord={:}): ValueError".format(ord))
                    with self.assertRaises(ValueError):
                        array.norm(ord=ord)
                    continue

                # Scipy does not supports negative norms
                if array.issparse is True and is_int(ord) and ord < 0:
                    self.logger.info(
                        "array.norm(ord={:}): ValueError".format(ord))
                    with self.assertRaises(NotImplementedError):
                        array.norm(ord=ord)
                    continue

                res = array.norm(ord=ord)

                self.logger.info("array.norm(ord={:}):\n{:}"
                                 "".format(ord, res))

                # Special handle of empty arrays
                if array.size == 0:
                    self.assertTrue(is_scalar(res))
                    self.assertEqual(float, type(res))
                    self.assertEqual(0, res)
                    continue

                res_np = np.linalg.norm(
                    array.tondarray().ravel(), ord=ord).round(4)

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

        def _check_norm_2d(array):
            self.logger.info("array:\n{:}".format(array))

            for axis_idx, axis in enumerate((None, 0, 1)):
                for ord_idx, ord in enumerate(
                        (None, 'fro', np.inf, -np.inf, 1, -1, 2, -2, 3, -3)):

                    if axis is None and ord in (2, -2):
                        self.logger.info(
                            "array.norm_2d(ord={:}, axis={:}): "
                            "NotImplementedError".format(ord, axis))
                        # Norms not implemented for matrices
                        with self.assertRaises(NotImplementedError):
                            array.norm_2d(ord=ord, axis=axis)
                        continue

                    if axis is None and ord in (3, -3):
                        self.logger.info(
                            "array.norm_2d(ord={:}, axis={:}): "
                            "ValueError".format(ord, axis))
                        # Invalid norm order for matrices
                        with self.assertRaises(ValueError):
                            array.norm_2d(ord=ord, axis=axis)
                        continue

                    if axis is not None and ord == 'fro':
                        self.logger.info(
                            "array.norm_2d(ord={:}, axis={:}): "
                            "ValueError".format(ord, axis))
                        # fro-norm is a matrix norm
                        with self.assertRaises(ValueError):
                            array.norm_2d(ord=ord, axis=axis)
                        continue

                    if array.issparse is True and axis is not None and \
                            (is_int(ord) and ord < 0):
                        self.logger.info(
                            "array.norm_2d(ord={:}, axis={:}): "
                            "NotImplementedError".format(ord, axis))
                        # Negative vector norms not implemented for sparse
                        with self.assertRaises(NotImplementedError):
                            array.norm_2d(ord=ord, axis=axis)
                        continue

                    res = array.norm_2d(ord=ord, axis=axis)
                    self.logger.info("array.norm_2d(ord={:}, axis={:}):"
                                     "\n{:}".format(ord, axis, res))

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
                                            ord=ord, axis=axis,
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
                    self.logger.info("array.norm_2d(ord={:}): "
                                     "NotImplementedError".format(0))
                    array.norm_2d(ord=0)  # Norm 0 not implemented

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

    def test_mixed(self):
        """Used for mixed testing.
        * * * * DO NOT DELETE * * * *
        """

        pass
    

if __name__ == '__main__':
    CUnitTest.main()
