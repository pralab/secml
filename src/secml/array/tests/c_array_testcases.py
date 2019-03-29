from secml.testing import CUnitTest
from secml.array import CArray


class CArrayTestCases(CUnitTest):
    """Unittests interface for CArray."""

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

        self.single_flat_dense_zero = CArray([0])
        self.single_dense_zero = self.single_flat_dense_zero.atleast_2d()
        self.single_sparse_zero = CArray(
            self.single_dense_zero.deepcopy(), tosparse=True)

        self.single_bool_flat_dense = CArray([True])
        self.single_bool_dense = self.single_bool_flat_dense.atleast_2d()
        self.single_bool_sparse = CArray(
            self.single_bool_dense.deepcopy(), tosparse=True)

        self.single_bool_flat_dense_false = CArray([False])
        self.single_bool_dense_false = \
            self.single_bool_flat_dense_false.atleast_2d()
        self.single_bool_sparse_false = CArray(
            self.single_bool_dense_false.deepcopy(), tosparse=True)

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
            self.assert_array_equal(item, items_list[item_idx + 1])

        # Every item is equal to each other, return True
        return True

    def _test_operator_cycle(self, totest_op, totest_items, totest_result):
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

    def _test_operator_notimplemented(self, totest_op, totest_items):
        """Check if operator is not implemented for given items.

        totest_op: list of operators
        totest_items: list of items PAIR to test

        """
        for operator in totest_op:
            for pair in totest_items:
                with self.assertRaises(NotImplementedError):
                    operator(pair[0], pair[1])
