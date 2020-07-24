from secml.array.tests import CArrayTestCases

import numpy as np
import itertools

from secml.array import CArray


class TestCArrayIndexing(CArrayTestCases):
    """Unit test for CArray INDEXING methods."""

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
                                 "{:} is different from {:}".format(
                                     selection, target_list[selector_idx]))

                if isinstance(target_list[selector_idx], CArray):
                    self.assertIsInstance(selection, CArray)
                    if selection.issparse:
                        self.assertEqual(
                            target_list[selector_idx].atleast_2d().shape,
                            selection.shape)
                    else:
                        self.assertEqual(target_list[selector_idx].shape,
                                         selection.shape)
                    self.assertEqual(
                        target_list[selector_idx].dtype, selection.dtype)
                else:
                    self.assertIsInstance(
                        selection, type(target_list[selector_idx]))

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
                                CArray([[2, 2]]), CArray([2, 2], tosparse=True),
                                slice(1, 3), [False, True, True],
                                CArray([False, True, True])]
            selectors = list(itertools.product(selectors_unique, repeat=2)) + \
                [(1, slice(None, None, -1)), (slice(None, None, -1), 1),
                 ([1, 1], slice(None, None, -1)), (slice(None, None, -1), [1, 1])]

            targets_a = [CArray([[0]]), CArray([[0]]),   # 2
                         CArray([[0, 0]]), CArray([[0, 0]]),
                         CArray([[0, 0]]), CArray([[0, 0]]),
                         CArray([[6, 0]]), CArray([[6, 0]]), CArray([[6, 0]])
                         ]
            targets_b = [CArray([[0], [0]]), CArray([[0], [0]]),   # [2, 2]
                         CArray([[0, 0], [0, 0]]), CArray([[0, 0], [0, 0]]),
                         CArray([[0, 0], [0, 0]]), CArray([[0, 0], [0, 0]]),
                         CArray([[6, 0], [6, 0]]),
                         CArray([[6, 0], [6, 0]]), CArray([[6, 0], [6, 0]])
                         ]
            targets_c = [CArray([[6], [0]]), CArray([[6], [0]]),   # [False, True, True]
                         CArray([[6, 6], [0, 0]]), CArray([[6, 6], [0, 0]]),
                         CArray([[6, 6], [0, 0]]), CArray([[6, 6], [0, 0]]),
                         CArray([[4, 6], [6, 0]]),
                         CArray([[4, 6], [6, 0]]), CArray([[4, 6], [6, 0]])
                         ]
            targets_d = [CArray([[6, 4, 2]]), CArray([[6], [4], [2]]),
                         CArray([[6, 4, 2], [6, 4, 2]]),
                         CArray([[6, 6], [4, 4], [2, 2]])]

            targets = 2 * targets_a + 4 * targets_b + 3 * targets_c + targets_d

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
                           CArray([True, False, True]),
                           slice(None, None, -1)
                           ]
            selectors_row = [0, np.ravel(0)[0], [0], CArray([0]),
                             -1, np.ravel(-1)[0], [-1], CArray([-1]),
                             True, np.ravel(True)[0], [True], CArray([True])]
            selectors_col = [[], 0, np.ravel(0)[0], [2, 2], CArray([2, 2]),
                             slice(1, 3), [False, True, True],
                             CArray([False, True, True])]
            selectors_b = [(0, slice(None, None, -1)),
                           (CArray([0]), slice(None, None, -1))]
            selectors = selectors_a + [(x, y) for x in selectors_row for y in selectors_col] + \
                        selectors_b

            targets_a = [CArray([6, 4, 0])]
            targets_b = [CArray([4, 6])]
            targets = 4 * targets_a + 2 * targets_b
            # Output always flat for flat arrays
            if array.ndim == 1:
                targets += [CArray([6, 0, 4])]
                targets += 12 * ([CArray([], dtype=int)] + 2 * [CArray([4])] +
                                 2 * [CArray([6, 6])] + 3 * [CArray([0, 6])])
                targets += 2 * [CArray([6, 0, 4])]
            else:
                targets += [CArray([[6, 0, 4]])]
                targets += 12 * ([CArray([[]], dtype=int)] + 2 * [CArray([[4]])] +
                                 2 * [CArray([[6, 6]])] + 3 * [CArray([[0, 6]])])
                targets += 2 * [CArray([[6, 0, 4]])]

            test_selectors(array, selectors, targets)

        # 1D INDEXING (VECTOR-LIKE)
        arrays_list = [self.row_flat_dense, self.row_dense, self.row_sparse]
        for array in arrays_list:

            self.logger.info("Testing getters for vector: \n" + str(array))

            selectors = [[], 0, np.ravel(0)[0], [2, 2], CArray([2, 2]),
                         slice(1, 3), slice(None), slice(None, None, -1)]

            # Output always flat for flat arrays
            if array.ndim == 1:
                targets = [CArray([], dtype=int)] + 2 * [CArray([4])] + \
                          2 * [CArray([6, 6])] + [CArray([0, 6])] + \
                          [CArray([4, 0, 6])] + [CArray([6, 0, 4])]
            else:
                targets = [CArray([[]], dtype=int)] + 2 * [CArray([[4]])] + \
                          2 * [CArray([[6, 6]])] + [CArray([[0, 6]])] + \
                          [CArray([[4, 0, 6]])] + [CArray([[6, 0, 4]])]

            test_selectors(array, selectors, targets)

        # SPECIAL CASE: SIZE 1 ARRAY
        arrays_list = [self.single_flat_dense, self.single_dense, self.single_sparse]
        for array in arrays_list:

            self.logger.info("Testing getters for array: \n" + str(array))

            selectors = [[], 0, np.ravel(0)[0], True, [True], CArray([True]),
                         slice(0, 1), slice(None), slice(None, None, -1), CArray([0, 0])]

            # CArray([True]) is considered a boolean mask in this case,
            # resulting selection is always flat
            if array.ndim == 1:
                targets = [CArray([], dtype=int)] + 4 * [CArray([4])] + \
                          [CArray([4])] + 3 * [CArray([4])] + [CArray([4, 4])]
            else:
                targets = [CArray([[]], dtype=int)] + 4 * [CArray([[4]])] + \
                          [CArray([4])] + 3 * [CArray([[4]])] + [CArray([[4, 4]])]

            test_selectors(array, selectors, targets)

    def test_setter(self):
        """Method that tests __setitem__ methods."""

        def test_selectors(input_array, selector_list, assignment_list, target_list):

            for selector_idx, selector in enumerate(selector_list):

                self.logger.info("Set: array[{:}] = {:}".format(
                    selector, assignment_list[selector_idx]))
                array_copy = input_array.deepcopy()
                try:  # Using a try to easier debug
                    array_copy[selector] = assignment_list[selector_idx]
                except (IndexError, ValueError, TypeError):
                    array_copy[selector] = assignment_list[selector_idx]
                self.logger.info("Result is: \n" + str(array_copy))

                self.assertFalse(CArray(array_copy != target_list[selector_idx]).any(),
                                 "{:} is different from {:}".format(
                                     array_copy, target_list[selector_idx]))

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

            selectors_unique = [2, np.ravel(2)[0],
                                [1, 2], CArray([1, 2]),
                                CArray([[1, 2]]), CArray([1, 2], tosparse=True),
                                slice(1, 3), [False, True, True],
                                CArray([False, True, True])]
            selectors = list(itertools.product(selectors_unique, repeat=2))
            selectors += [(2, 2), (2, 2)]

            assignments_a = [10, 10] + 5 * [CArray([[10, 20]])] + 2 * [CArray([10, 20])]
            assignments_b = [CArray([[10], [20]])] + [CArray([[10], [20]], tosparse=True)] + \
                             7 * [CArray([[10, 20], [30, 40]])]
            assignments_c = [CArray([10]), CArray([10], tosparse=True)]
            assignments = 2 * assignments_a + 7 * assignments_b + assignments_c

            targets_a = 2 * [CArray([[1, 2, 0], [2, 4, 6], [0, 6, 10]])] + \
                        7 * [CArray([[1, 2, 0], [2, 4, 6], [0, 10, 20]])]
            targets_b = 2 * [CArray([[1, 2, 0], [2, 4, 10], [0, 6, 20]])] + \
                        7 * [CArray([[1, 2, 0], [2, 10, 20], [0, 30, 40]])]
            targets_c = 2 * [CArray([[1, 2, 0], [2, 4, 6], [0, 6, 10]])]
            targets = 2 * targets_a + 7 * targets_b + targets_c

            test_selectors(array, selectors, assignments, targets)

        # 2D/1D INDEXING (VECTOR-LIKE)
        arrays_list = [self.row_flat_dense, self.row_dense, self.row_sparse]
        for array in arrays_list:

            self.logger.info("Testing setters for array: \n" + str(array))

            selectors_a = [[[0, 0], [2, 0]],
                           [[0, 0], [np.ravel(2)[0], np.ravel(0)[0]]],
                           [[np.ravel(0)[0], np.ravel(0)[0]], [2, 0]],
                           [[np.ravel(0)[0], np.ravel(0)[0]], [np.ravel(2)[0], np.ravel(0)[0]]],
                           CArray([[True, False, True]]), CArray([True, False, True]),
                           CArray([[True, False, True]]), CArray([True, False, True])
                           ]
            selectors_row = [0, np.ravel(0)[0], [0], CArray([0]),
                             -1, np.ravel(-1)[0], [-1], CArray([-1]),
                             True, np.ravel(True)[0], [True], CArray([True])]
            selectors_col = [[], 0, np.ravel(0)[0], [1, 2], CArray([1, 2]),
                             slice(1, 3), [False, True, True], CArray([False, True, True])]
            selectors = selectors_a + [(x, y) for x in selectors_row for y in selectors_col]

            assignments_a = 2 * [CArray([10, 20])] + \
                            2 * [CArray([[10, 20]])] + \
                            2 * [CArray([10, 20])] + \
                            2 * [CArray([[10, 20]])]
            assignments_b = [0] + [10, 10] + \
                            2 * [CArray([[10, 20]])] + \
                            3 * [CArray([10, 20])]
            assignments = assignments_a + 12 * assignments_b

            targets_a = CArray([20, 0, 10])
            targets_b = CArray([10, 0, 20])
            targets_c = CArray([10, 0, 6])
            targets_d = CArray([4, 10, 20])
            # Output always flat for flat arrays
            if array.ndim == 1:
                targets = 4 * [targets_a] + 4 * [targets_b] + \
                          12 * ([CArray([4, 0, 6])] + 2 * [targets_c] + 5 * [targets_d])
            else:
                targets = 4 * [targets_a.atleast_2d()] + 4 * [targets_b.atleast_2d()] + \
                            12 * ([CArray([[4, 0, 6]])] +
                                  2 * [targets_c.atleast_2d()] + 5 * [targets_d.atleast_2d()])

            test_selectors(array, selectors, assignments, targets)

        # 1D INDEXING (VECTOR-LIKE)
        arrays_list = [self.row_flat_dense, self.row_dense, self.row_sparse]
        for array in arrays_list:

            self.logger.info("Testing setters for vector: \n" + str(array))

            selectors = [[], 0, np.ravel(0)[0], [1, 2], CArray([1, 2]),
                         slice(1, 3), slice(None), 0, 0]

            assignments = [0] + [10, 10] + 2 * [CArray([[10, 20]])] + \
                          [CArray([[10, 20]], tosparse=True)] + [CArray([[10, 20, 30]])] + \
                          [CArray([10]), CArray([10], tosparse=True)]

            targets_a = CArray([10, 0, 6])
            targets_b = CArray([4, 10, 20])
            targets_c = CArray([10, 20, 30])
            # Output always flat for flat arrays
            if array.ndim == 1:
                targets = [CArray([4, 0, 6])] + 2 * [targets_a] + \
                          3 * [targets_b] + [targets_c] + 2 * [targets_a]
            else:
                targets = [CArray([[4, 0, 6]])] + 2 * [targets_a.atleast_2d()] + \
                          3 * [targets_b.atleast_2d()] + [targets_c.atleast_2d()] + \
                          2 * [targets_a.atleast_2d()]

            test_selectors(array, selectors, assignments, targets)

        # SPECIAL CASE: SIZE 1 ARRAY
        arrays_list = [self.single_flat_dense, self.single_dense, self.single_sparse]
        for array in arrays_list:

            self.logger.info("Testing setters for array: \n" + str(array))

            selectors = [[], 0, np.ravel(0)[0], True, [True], CArray([True]),
                         slice(0, 1), slice(None), 0, 0]

            assignments = 8 * [10] + [CArray([10]), CArray([10], tosparse=True)]

            targets_a = CArray([10])
            # Output always flat for flat arrays
            if array.ndim == 1:
                targets = [CArray([4])] + 9 * [targets_a]
            else:
                targets = [CArray([[4]])] + 9 * [targets_a.atleast_2d()]

            test_selectors(array, selectors, assignments, targets)

    def test_item(self):
        """Test for CArray.item() method."""
        self.logger.info("Test for CArray.item() method")

        def _item(a):
            x = a.item()
            self.logger.info("array:\n{:}\nextracted: {:}".format(a, x))
            self.assertEqual(a.dtype, type(x))

        _item(self.single_flat_dense.astype(int))
        _item(self.single_dense.astype(int))
        _item(self.single_sparse.astype(int))
        _item(self.single_flat_dense.astype(float))
        _item(self.single_dense.astype(float))
        _item(self.single_sparse.astype(float))
        _item(self.single_bool_flat_dense)
        _item(self.single_bool_dense)
        _item(self.single_bool_sparse)

        with self.assertRaises(ValueError):
            self.empty_flat_dense.item()
        with self.assertRaises(ValueError):
            self.empty_dense.item()
        with self.assertRaises(ValueError):
            self.empty_sparse.item()

        with self.assertRaises(ValueError):
            self.array_dense.item()
        with self.assertRaises(ValueError):
            self.array_sparse.item()
    

if __name__ == '__main__':
    CArrayTestCases.main()
