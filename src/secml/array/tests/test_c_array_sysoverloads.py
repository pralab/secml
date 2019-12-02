from secml.array.tests import CArrayTestCases

import numpy as np
import scipy.sparse as scs
import operator as op
from itertools import product

from secml.array import CArray
from secml.array.c_dense import CDense
from secml.array.c_sparse import CSparse


class TestCArraySystemOverloads(CArrayTestCases):
    """Unit test for CArray SYSTEM OVERLOADS methods."""

    def test_operators_array_vs_array_broadcast(self):
        """Test for mathematical operators array vs array with broadcast."""
        operators = [op.add, op.sub]
        expected_result = [CSparse, CDense, CDense, CDense]
        items = [(self.array_sparse_sym, self.row_sparse),
                 (self.array_sparse_sym, self.row_dense),
                 (self.array_dense_sym, self.row_sparse),
                 (self.array_dense_sym, self.row_dense)]
        self._test_operator_cycle(operators, items, expected_result)

        operators = [op.mul]
        expected_result = [CSparse, CSparse, CSparse, CDense]
        items = [(self.array_sparse_sym, self.row_sparse),
                 (self.array_sparse_sym, self.row_dense),
                 (self.array_dense_sym, self.row_sparse),
                 (self.array_dense_sym, self.row_dense)]
        self._test_operator_cycle(operators, items, expected_result)

        operators = [op.truediv, op.floordiv]
        expected_result = [CDense, CDense, CDense, CDense]
        items = [(self.array_sparse_sym, self.row_sparse),
                 (self.array_sparse_sym, self.row_dense),
                 (self.array_dense_sym, self.row_sparse),
                 (self.array_dense_sym, self.row_dense)]

        with self.logger.catch_warnings():
            # For 0 / 0 divisions
            self.logger.filterwarnings(
                action='ignore',
                message="divide by zero encountered in true_divide",
                category=RuntimeWarning)
            self._test_operator_cycle(operators, items, expected_result)

        operators = [op.pow, CArray.pow]
        expected_result = [CDense, CDense]
        items = [(self.array_dense_sym, self.row_sparse),
                 (self.array_dense_sym, self.row_dense)]
        self._test_operator_cycle(operators, items, expected_result)

        # Sparse array ** array is not supported
        with self.assertRaises(TypeError):
            self.array_sparse ** self.row_sparse
        with self.assertRaises(TypeError):
            self.array_sparse ** self.row_dense
        with self.assertRaises(TypeError):
            self.array_sparse.pow(self.row_sparse)
        with self.assertRaises(TypeError):
            self.array_sparse.pow(self.row_dense)

    def test_operators_array_vs_array(self):
        """Test for mathematical operators array vs array."""
        operators = [op.add, op.sub]
        expected_result = [CSparse, CDense, CDense, CDense]
        items = [(self.array_sparse, self.array_sparse),
                 (self.array_sparse, self.array_dense),
                 (self.array_dense, self.array_sparse),
                 (self.array_dense, self.array_dense)]
        self._test_operator_cycle(operators, items, expected_result)

        operators = [op.mul]
        expected_result = [CSparse, CSparse, CSparse, CDense]
        items = [(self.array_sparse, self.array_sparse),
                 (self.array_sparse, self.array_dense),
                 (self.array_dense, self.array_sparse),
                 (self.array_dense, self.array_dense)]
        self._test_operator_cycle(operators, items, expected_result)

        operators = [op.truediv, op.floordiv]
        expected_result = [CDense, CDense, CDense, CDense]
        items = [(self.array_sparse, self.array_sparse),
                 (self.array_sparse, self.array_dense),
                 (self.array_dense, self.array_sparse),
                 (self.array_dense, self.array_dense)]

        with self.logger.catch_warnings():
            # For 0 / 0 divisions
            self.logger.filterwarnings(
                action='ignore',
                message="invalid value encountered in true_divide",
                category=RuntimeWarning)
            self._test_operator_cycle(operators, items, expected_result)

        operators = [op.pow, CArray.pow]
        expected_result = [CDense, CDense]
        items = [(self.array_dense, self.array_sparse),
                 (self.array_dense, self.array_dense)]
        self._test_operator_cycle(operators, items, expected_result)

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
        self.assertIsInstance(s_abs._data, CSparse)
        self.assertIsInstance(d_abs._data, CDense)
        # Check if we have the same output in all cases
        self.assertTrue(self._test_multiple_eq([s_abs, d_abs]))

        # array.abs()
        self.logger.info("Checking .abs() method...")
        s_abs = self.array_sparse.abs()
        d_abs = self.array_dense.abs()
        # Check if method returned correct datatypes
        self.assertIsInstance(s_abs._data, CSparse)
        self.assertIsInstance(d_abs._data, CDense)
        # Check if we have the same output in all cases
        self.assertTrue(self._test_multiple_eq([s_abs, d_abs]))

        # Negative
        self.logger.info("Checking negative operator...")
        s_abs = -self.array_sparse
        d_abs = -self.array_dense
        # Check if method returned correct datatypes
        self.assertIsInstance(s_abs._data, CSparse)
        self.assertIsInstance(d_abs._data, CDense)
        # Check if we have the same output in all cases
        self.assertTrue(self._test_multiple_eq([s_abs, d_abs]))

    def test_operators_array_vs_scalar(self):
        """Test for mathematical operators array vs scalar."""

        test_scalars = [
            2, np.ravel(2)[0], 2.0, np.ravel(2.0)[0], np.float32(2.0)]
        test_z_scalars = [
            0, np.ravel(0)[0], 0.0, np.ravel(0.0)[0], np.float32(0.0)]

        # DENSE ARRAY + NONZERO SCALAR, NONZERO SCALAR + DENSE ARRAY
        # sparse array + nonzero scalar is not supported (and viceversa)
        operators = [op.add, op.mul]
        expected_result = [CDense] * 10
        items = list(product([self.array_dense], test_scalars)) + \
            list(product(test_scalars, [self.array_dense]))
        self._test_operator_cycle(operators, items, expected_result)

        # ARRAY + ZERO SCALAR, ZERO SCALAR + ARRAY
        operators = [op.add, op.mul]
        expected_result = [CDense] * 10 + [CSparse] * 10
        items = list(product([self.array_dense], test_z_scalars)) + \
            list(product(test_z_scalars, [self.array_dense])) + \
            list(product([self.array_sparse], test_z_scalars)) + \
            list(product(test_z_scalars, [self.array_sparse]))
        self._test_operator_cycle(operators, items, expected_result)

        # DENSE ARRAY - NONZERO SCALAR
        # sparse array - nonzero scalar is not supported (and viceversa)
        operators = [op.sub]
        expected_result = [CDense] * 5
        items = list(product([self.array_dense], test_scalars))
        self._test_operator_cycle(operators, items, expected_result)

        # NONZERO SCALAR - DENSE ARRAY
        operators = [op.sub]
        expected_result = [CDense] * 5
        items = list(product(test_scalars, [self.array_dense]))
        self._test_operator_cycle(operators, items, expected_result)

        # ARRAY - ZERO SCALAR
        operators = [op.sub]
        expected_result = [CDense] * 5 + [CSparse] * 5
        items = list(product([self.array_dense], test_z_scalars)) + \
            list(product([self.array_sparse], test_z_scalars))
        self._test_operator_cycle(operators, items, expected_result)

        # ZERO SCALAR - ARRAY
        operators = [op.sub]
        expected_result = [CDense] * 5 + [CSparse] * 5
        items = list(product(test_z_scalars, [self.array_dense])) + \
            list(product(test_z_scalars, [self.array_sparse]))
        self._test_operator_cycle(operators, items, expected_result)

        # ARRAY * NONZERO SCALAR, NONZERO SCALAR * ARRAY
        operators = [op.mul]
        expected_result = [CDense] * 10 + [CSparse] * 10
        items = list(product([self.array_dense], test_scalars)) + \
            list(product(test_scalars, [self.array_dense])) + \
            list(product([self.array_sparse], test_scalars)) + \
            list(product(test_scalars, [self.array_sparse]))
        self._test_operator_cycle(operators, items, expected_result)

        # ARRAY * ZERO SCALAR, ZERO SCALAR * ARRAY
        operators = [op.mul]
        expected_result = [CDense] * 10 + [CSparse] * 10
        items = list(product([self.array_dense], test_z_scalars)) + \
            list(product(test_z_scalars, [self.array_dense])) + \
            list(product([self.array_sparse], test_z_scalars)) + \
            list(product(test_z_scalars, [self.array_sparse]))
        self._test_operator_cycle(operators, items, expected_result)

        # ARRAY / NONZERO SCALAR
        operators = [op.truediv, op.floordiv]
        expected_result = [CDense] * 5 + [CSparse] * 5
        items = list(product([self.array_dense], test_scalars)) + \
            list(product([self.array_sparse], test_scalars))
        self._test_operator_cycle(operators, items, expected_result)

        # NONZERO SCALAR / DENSE ARRAY
        # nonzero scalar / sparse array is not supported
        operators = [op.truediv, op.floordiv]
        expected_result = [CDense] * 5
        items = list(product(test_scalars, [self.array_dense]))
        with self.logger.catch_warnings():
            # we are dividing using arrays having zeros
            self.logger.filterwarnings(
                action='ignore',
                message="divide by zero encountered in true_divide",
                category=RuntimeWarning)
            self.logger.filterwarnings(
                action='ignore',
                message="divide by zero encountered in divide",
                category=RuntimeWarning)
            self._test_operator_cycle(operators, items, expected_result)

        # ZERO SCALAR / DENSE ARRAY
        # zero scalar / sparse array is not supported
        operators = [op.truediv, op.floordiv]
        expected_result = [CDense] * 5
        items = list(product(test_z_scalars, [self.array_dense]))
        with self.logger.catch_warnings():
            # we are dividing a zero scalar by something
            self.logger.filterwarnings(
                action='ignore',
                message="divide by zero encountered in true_divide",
                category=RuntimeWarning)
            # For 0 / 0 divisions
            self.logger.filterwarnings(
                action='ignore',
                message="invalid value encountered in true_divide",
                category=RuntimeWarning)
            self.logger.filterwarnings(
                action='ignore',
                message="invalid value encountered in divide",
                category=RuntimeWarning)
            self._test_operator_cycle(operators, items, expected_result)

        # ARRAY ** NONZERO SCALAR
        operators = [op.pow, CArray.pow]
        expected_result = [CDense] * 5 + [CSparse] * 5
        items = list(product([self.array_dense], test_scalars)) + \
            list(product([self.array_sparse], test_scalars))
        self._test_operator_cycle(operators, items, expected_result)

        # NONZERO SCALAR ** DENSE ARRAY
        # nonzero scalar ** sparse array is not supported
        operators = [op.pow]
        expected_result = [CDense] * 5
        items = list(product(test_scalars, [self.array_dense]))
        self._test_operator_cycle(operators, items, expected_result)

        # DENSE ARRAY ** ZERO SCALAR
        # sparse array ** zero scalar is not supported
        operators = [op.pow, CArray.pow]
        expected_result = [CDense] * 5
        items = list(product([self.array_dense], test_z_scalars))
        self._test_operator_cycle(operators, items, expected_result)

        # ZERO SCALAR ** DENSE ARRAY
        # zero scalar ** sparse array is not supported
        operators = [op.pow]
        expected_result = [CDense] * 5
        items = list(product(test_z_scalars, [self.array_dense]))
        self._test_operator_cycle(operators, items, expected_result)

        # NONZERO SCALAR +,- SPARSE ARRAY NOT SUPPORTED (AND VICEVERSA)
        items = list(product([self.array_sparse], test_scalars)) + \
            list(product(test_scalars, [self.array_sparse]))
        operators = [op.add, op.sub]
        self._test_operator_notimplemented(operators, items)

        # ZERO SCALAR / SPARSE ARRAY NOT SUPPORTED
        # NONZERO SCALAR / SPARSE ARRAY NOT SUPPORTED
        items = list(product(test_scalars, [self.array_sparse])) + \
            list(product(test_z_scalars, [self.array_sparse]))
        operators = [op.truediv, op.floordiv]
        self._test_operator_notimplemented(operators, items)

        # NONZERO SCALAR ** SPARSE ARRAY NOT SUPPORTED
        # ZERO SCALAR ** SPARSE ARRAY NOT SUPPORTED
        items = list(product(test_scalars, [self.array_sparse])) + \
            list(product(test_z_scalars, [self.array_sparse]))
        operators = [op.pow]
        self._test_operator_notimplemented(operators, items)

        # SPARSE ARRAY ** ZERO SCALAR NOT SUPPORTED
        items = list(product([self.array_sparse], test_z_scalars))
        operators = [op.pow]
        self._test_operator_notimplemented(operators, items)

        # TODO: ARRAY / ZERO SCALAR TEST (SEE #353)

    def test_operators_array_vs_unsupported(self):
        """Test for mathematical operators array vs unsupported types."""

        def test_unsupported(x):
            operators = [op.add, op.sub, op.mul,
                         op.truediv, op.floordiv, op.pow]
            for operator in operators:
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
            operators = [op.add, op.sub, op.mul,
                         op.truediv, op.floordiv, op.pow]
            for operator in operators:
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
        expected_result = [CSparse, CDense, CDense, CDense]
        items = [(self.array_sparse, self.array_sparse),
                 (self.array_sparse, self.array_dense),
                 (self.array_dense, self.array_sparse),
                 (self.array_dense, self.array_dense)]

        with self.logger.catch_warnings():
            # Comparing sparse arrays using ==, <= and >= is inefficient
            self.logger.filterwarnings(
                action='ignore',
                message="Comparing sparse matrices using*",
                category=scs.SparseEfficiencyWarning)
            self._test_operator_cycle(operators, items, expected_result)

    def test_comparison_array_vs_array_broadcast(self):
        """Test for comparison operators array vs array with broadcast."""
        operators = [op.eq, op.lt, op.le, op.gt, op.ge, op.ne]
        expected_result = [CSparse, CDense, CDense, CDense]
        items = [(self.array_sparse_sym, self.row_sparse),
                 (self.array_sparse_sym, self.row_dense),
                 (self.array_dense_sym, self.row_sparse),
                 (self.array_dense_sym, self.row_dense)]

        with self.logger.catch_warnings():
            # Comparing sparse arrays using ==, <= and >= is inefficient
            self.logger.filterwarnings(
                action='ignore',
                message="Comparing sparse matrices using*",
                category=scs.SparseEfficiencyWarning)
            self._test_operator_cycle(operators, items, expected_result)

    def test_comparison_array_vs_scalar(self):
        """Test for comparison operators array vs scalar."""
        operators = [op.eq, op.lt, op.le, op.gt, op.ge, op.ne]
        expected_result = [CSparse, CDense, CSparse, CDense]
        items = [(self.array_sparse, 2),
                 (self.array_dense, 2),
                 (self.array_sparse, np.ravel(2)[0]),
                 (self.array_dense, np.ravel(2)[0])]
        with self.logger.catch_warnings():
            # Comparing a sparse matrix with a scalar greater than zero
            # using < or <= is inefficient
            # Comparing a sparse matrix with a nonzero scalar
            # using != is inefficient
            self.logger.filterwarnings(
                action='ignore',
                message="Comparing a sparse matrix*",
                category=scs.SparseEfficiencyWarning)
            self._test_operator_cycle(operators, items, expected_result)

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
            print(d and c)
        with self.assertRaises(ValueError):
            print(c and d)
        with self.assertRaises(ValueError):
            print(d or c)
        with self.assertRaises(ValueError):
            print(c or d)

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
            self.assertEqual(self.array_dense.ravel()[elem_id].item(),  elem)
        # Check if all array elements have been returned
        self.assertEqual(self.array_dense.size, len(res))

        res = []
        for elem_id, elem in enumerate(self.array_sparse):
            res.append(elem)
            self.assertEqual(self.array_sparse.ravel()[elem_id].item(), elem)
        # Check if all array elements have been returned
        self.assertEqual(self.array_sparse.size, len(res))

        res = []
        for elem_id, elem in enumerate(self.row_flat_dense):
            res.append(elem)
            self.assertEqual(self.row_flat_dense[elem_id].item(), elem)
        # Check if all array elements have been returned
        self.assertEqual(self.row_flat_dense.size, len(res))

        res = []
        for elem_id, elem in enumerate(self.row_dense):
            res.append(elem)
            self.assertEqual(self.row_dense[elem_id].item(), elem)
        # Check if all array elements have been returned
        self.assertEqual(self.row_dense.size, len(res))

        res = []
        for elem_id, elem in enumerate(self.row_sparse):
            res.append(elem)
            self.assertEqual(self.row_sparse[elem_id].item(), elem)
        # Check if all array elements have been returned
        self.assertEqual(self.row_sparse.size, len(res))


if __name__ == '__main__':
    CArrayTestCases.main()
