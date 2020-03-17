from secml.testing import CUnitTest

from secml.array import CArray
from secml.data import CDatasetHeader


class TestCDatasetHeader(CUnitTest):
    """Unittest for CDatasetHeader"""

    def setUp(self):

        self.header = CDatasetHeader(
            id='mydataset', age=34,
            colors=CArray([1, 2, 3]),
            days=('Mon', 'Tue', 'Wed', 'Thu'))

    def test_properties(self):
        """Test class properties."""
        self.assertEqual('mydataset', self.header.id)
        self.logger.info("header.id: {:}".format(self.header.id))

        self.assertEqual(34, self.header.age)
        self.logger.info("header.age: {:}".format(self.header.age))

        self.assertEqual(('Mon', 'Tue', 'Wed', 'Thu'), self.header.days)
        self.logger.info("header.days: {:}".format(self.header.days))

        self.assert_array_equal(CArray([1, 2, 3]), self.header.colors)
        self.logger.info("header.colors: {:}".format(self.header.colors))

    def test_additional_attr(self):
        """Test for adding new attributed to header."""
        self.header.b = 3
        self.assertEqual(3, self.header.b)
        self.logger.info("header.b: {:}".format(self.header.b))

        self.header.c = [1, 2, 3]  # Will be converted to CArray
        self.assertIsInstance(self.header.c, CArray)
        self.assert_array_equal(CArray([1, 2, 3]), self.header.c)
        self.logger.info("header.c: {:}".format(self.header.c))

        # A new CArray could be set only if has a compatible number of
        # samples, which is 3 after 'colors'
        with self.assertRaises(ValueError):
            self.header.d = CArray([1, 2, 3, 4])

    def test_getitem(self):
        """Test for getter."""
        h_get = self.header[[0, 2]]
        params = h_get.get_params()
        self.assertEqual(params['id'], 'mydataset')
        self.assertEqual(params['age'], 34)
        self.assertEqual(params['days'], ('Mon', 'Tue', 'Wed', 'Thu'))
        self.assert_array_equal(
            params['colors'], CArray([1, 3]))

        with self.assertRaises(IndexError):  # 'colors' CArray has size 3
            self.header[[0, 3]]

    def test_append(self):
        """Test for .append() method."""

        h_append = self.header.append(self.header)
        params = h_append.get_params()
        self.assertEqual(params['id'], 'mydataset')
        self.assertEqual(params['age'], 34)
        self.assertEqual(params['days'], ('Mon', 'Tue', 'Wed', 'Thu'))
        self.assert_array_equal(
            params['colors'], CArray([1, 2, 3, 1, 2, 3]))

        # Create an additional header with new attributes set
        h2 = CDatasetHeader(
            a=4, age=34, colors=CArray([10, 20, 30])
        )

        h_append = self.header.append(h2)
        params = h_append.get_params()
        self.assertEqual(params['id'], 'mydataset')
        self.assertEqual(params['age'], 34)
        self.assertEqual(params['days'], ('Mon', 'Tue', 'Wed', 'Thu'))
        self.assertEqual(params['a'], 4)
        self.assert_array_equal(
            params['colors'], CArray([1, 2, 3, 10, 20, 30]))

    def test_copy(self):
        """Test for .deepcopy() method."""
        h_copy = self.header.deepcopy()

        # Now change original header
        self.header.colors[0] = 100
        params = self.header.get_params()
        self.assertEqual(params['id'], 'mydataset')
        self.assertEqual(params['age'], 34)
        self.assertEqual(params['days'], ('Mon', 'Tue', 'Wed', 'Thu'))
        self.assert_array_equal(params['colors'], CArray([100, 2, 3]))

        params = h_copy.get_params()
        self.assertEqual(params['id'], 'mydataset')
        self.assertEqual(params['age'], 34)
        self.assertEqual(params['days'], ('Mon', 'Tue', 'Wed', 'Thu'))
        self.assert_array_equal(params['colors'], CArray([1, 2, 3]))


if __name__ == "__main__":
    CUnitTest.main()
