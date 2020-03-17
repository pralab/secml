from secml.array.tests import CArrayTestCases

from secml.array import CArray


class TestCArrayUtilsSearch(CArrayTestCases):
    """Unit test for CArray UTILS - SEARCH methods."""

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
        _check_binary_search(CArray([[1], [2.4], [3], [4.3]]))
        _check_binary_search(CArray([[1, 2.4], [3, 4.3]]))

        # Sparse arrays are not supported
        with self.assertRaises(NotImplementedError):
            self.array_sparse.binary_search(3)


if __name__ == '__main__':
    CArrayTestCases.main()
