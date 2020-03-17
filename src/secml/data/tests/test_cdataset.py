from secml.testing import CUnitTest

from secml.array import CArray
from secml.data import CDataset, CDatasetHeader


class TestDataset(CUnitTest):
    """Unit test for CDataset"""

    def setUp(self):
        """Basic set up."""
        self.X = CArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.Y = CArray([1, 2, 2])
        self.dataset = CDataset(self.X, self.Y)

    def test_properties(self):
        """Test class properties."""
        self.logger.info("Dataset Patterns: \n" + str(self.dataset.X))
        self.logger.info("Dataset Labels: \n" + str(self.dataset.Y))
        self.logger.info("Number of classes: \n" + str(self.dataset.num_classes))
        self.logger.info("Number of patterns: \n" + str(self.dataset.num_samples))
        self.logger.info("Number of features: \n" + str(self.dataset.num_features))
        self.logger.info("Testing dataset properties...")
        self.assertEqual(
            2, self.dataset.num_classes, "Wrong number of classes!")
        self.assertEqual(
            3, self.dataset.num_samples, "Wrong number of patterns!")
        self.assertEqual(
            3, self.dataset.num_features, "Wrong number of features!")

    def test_getters_and_setters(self):
        """Test for getters and setters of the class."""
        self.logger.info("Testing setters and getters for the dataset...")
        self.assertTrue(
            (self.dataset.X == self.X).all(),  "Wrong pattern extraction")
        self.assertTrue(
            (self.dataset.Y == self.Y).all(), "Wrong labels extraction")

        new_patterns = CArray([[1, 2], [3, 4], [5, 6]])
        self.logger.info(
            "Setting new patterns: \n" + str(new_patterns))
        self.dataset.X = new_patterns
        self.logger.info("Testing new patterns...")
        self.assertTrue(
            (self.dataset.X == new_patterns).all(), "Wrong patterns set!")

        with self.assertRaises(ValueError):
            new_patterns = CArray([[1, 2, 3], [4, 5, 6]])
            self.logger.info(
                "Setting less patterns than labels: \n" + str(new_patterns))
            self.dataset.X = new_patterns

        new_labels = CArray([11, 22, 33])
        self.logger.info("Setting new labels: \n" + str(new_labels))
        self.dataset.Y = new_labels
        self.logger.info("Testing new labels...")
        self.assertTrue(
            (self.dataset.Y == new_labels).all(), "Wrong labels extraction")

        with self.assertRaises(ValueError):
            new_labels = CArray([1, 2])
            self.logger.info(
                "Setting less labels than patterns: \n" + str(new_labels))
            self.dataset.Y = new_labels

    def test_select_patterns(self):
        """Tests for select patterns method."""
        self.logger.info("Testing pattern extraction...")
        patterns = self.dataset.X[0:2, :]
        target = CArray([[1, 2, 3], [4, 5, 6]])
        self.logger.info("Extracting patterns:\n{:}".format(patterns))
        self.logger.info("Targets:\n{:}".format(target))
        self.logger.info("Testing row extraction...")
        self.assert_array_equal(patterns, target)

    def test_subset(self):
        """Tests for subset method."""
        self.logger.info("Testing subsets...")
        subset_lists = [([0, 1], [0, 1]),
                        ([0, 2], slice(0, 3)),
                        (slice(0, 3), [0, 2])]
        x_targets = [CArray([[1, 2], [4, 5]]),
                     CArray([[1, 2, 3], [7, 8, 9]]),
                     CArray([[1, 3], [4, 6], [7, 9]])]
        y_targets = [CArray([1, 2]), CArray([1, 2]), CArray([1, 2, 2])]
        for row_cols, Xtarget, Ytarget in zip(subset_lists, x_targets, y_targets):
            rows = row_cols[0]
            cols = row_cols[1]
            subset = self.dataset[rows, cols]
            self.logger.info(
                "Testing Subset extraction with rows indices: " + str(rows) +
                " and columns indices: " + str(cols) + " \n" + str(subset.X) +
                " \n" + str(subset.Y))
            self.assert_array_equal(subset.X, Xtarget)
            self.assert_array_equal(subset.Y, Ytarget)

    def test_custom_attr(self):
        """Testing for custom attributes."""
        header = CDatasetHeader(
            id='mydataset', age=34, colors=CArray([1, 2, 3]))
        ds = CDataset(self.X, self.Y, header=header)

        ds_params = ds.header.get_params()
        self.assertEqual(ds_params['id'], 'mydataset')
        self.assertEqual(ds_params['age'], 34)
        self.assert_array_equal(ds_params['colors'], CArray([1, 2, 3]))

        # Testing getitem. Immutable objects should be copied as they are.
        # Arrays should be indexed.
        ds_get = ds[[0, 2], :]
        ds_params = ds_get.header.get_params()
        self.assert_array_equal(ds_get.X, CArray([[1, 2, 3], [7, 8, 9]]))
        self.assert_array_equal(ds_get.Y, CArray([1, 2]))
        self.assertEqual(ds_params['id'], 'mydataset')
        self.assertEqual(ds_params['age'], 34)
        self.assert_array_equal(ds_params['colors'], CArray([1, 3]))

    def test_append(self):
        """Test for .append() method."""
        ds_append = self.dataset.append(self.dataset)

        self.assertEqual(self.dataset.num_samples * 2, ds_append.num_samples)

        self.assert_array_equal(
            ds_append.X, CArray([[1, 2, 3], [4, 5, 6], [7, 8, 9],
                                 [1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        self.assert_array_equal(ds_append.Y, CArray([1, 2, 2, 1, 2, 2]))

        # Test append with header
        ds = self.dataset.deepcopy()

        # Test append with header in both ds
        header = CDatasetHeader(
            id='mydataset', age=34, colors=CArray([1, 2, 3]))

        ds.header = header

        # Test append with header in both ds
        ds_append = ds.append(ds)
        ds_params = ds_append.header.get_params()
        self.assertEqual(ds_params['id'], 'mydataset')
        self.assertEqual(ds_params['age'], 34)
        self.assert_array_equal(
            ds_params['colors'], CArray([1, 2, 3, 1, 2, 3]))

        # Create two copies now for later tests
        ds1 = self.dataset.deepcopy()
        ds2 = self.dataset.deepcopy()

        # For the following tests we cannot use CArrays as params. Use tuple
        header = CDatasetHeader(
            id='mydataset', age=34, colors=(1, 2, 3))
        ds1.header = header
        ds2.header = header

        # Test append with header in first ds
        ds_append = ds1.append(self.dataset)
        ds_params = ds_append.header.get_params()
        self.assertEqual(ds_params['id'], 'mydataset')
        self.assertEqual(ds_params['age'], 34)
        self.assertEqual(ds_params['colors'], (1, 2, 3))

        # Test append with header in second ds
        ds_append = self.dataset.append(ds2)
        ds_params = ds_append.header.get_params()
        self.assertEqual(ds_params['id'], 'mydataset')
        self.assertEqual(ds_params['age'], 34)
        self.assert_array_equal(ds_params['colors'], (1, 2, 3))

    def test_copy(self):
        """Test for .deepcopy() method."""
        ds_copy = self.dataset.deepcopy()
        ds_copy.X[0, :] = 100
        ds_copy.Y[0] = 100

        self.assert_array_equal(self.dataset.X[0, :], CArray([[1, 2, 3]]))
        self.assert_array_equal(self.dataset.Y[0], CArray([1]))

        self.assert_array_equal(ds_copy.X[0, :], CArray([[100, 100, 100]]))
        self.assert_array_equal(ds_copy.Y[0], CArray([100]))

        # Test deepcopy with header
        header = CDatasetHeader(
            id='mydataset', age=34, colors=CArray([1, 2, 3]))
        self.dataset.header = header

        ds_copy = self.dataset.deepcopy()

        # Now change header of original dataset
        self.dataset.header.colors[0] = 100
        ds_params = self.dataset.header.get_params()
        self.assertEqual(ds_params['id'], 'mydataset')
        self.assertEqual(ds_params['age'], 34)
        self.assert_array_equal(ds_params['colors'], CArray([100, 2, 3]))

        ds_params = ds_copy.header.get_params()
        self.assertEqual(ds_params['id'], 'mydataset')
        self.assertEqual(ds_params['age'], 34)
        self.assert_array_equal(ds_params['colors'], CArray([1, 2, 3]))

    def test_labels_binarize(self):
        """Unittests for `.get_labels_ovr` and `.get_labels_onehot`."""
        self.logger.info("Testing `CDataset.get_labels_onehot() method")
        onehot = self.dataset.get_labels_onehot()

        self.assertEqual(
            (self.dataset.num_labels, self.dataset.Y.max() + 1), onehot.shape)
        self.assertFalse((onehot != 0).logical_and(onehot != 1).any())
        self.assertIsSubDtype(onehot.dtype, int)

        self.logger.info("Testing `CDataset.get_labels_ovr() method")
        for y_pos in (0, 1, 2):
            ovr = self.dataset.get_labels_ovr(pos_label=y_pos)
            self.logger.info("{:}".format(ovr))

            self.assertIsSubDtype(onehot.dtype, int)

            if y_pos not in self.dataset.classes:
                self.assertFalse((ovr != 0).any())
            else:
                self.assertTrue((ovr == 1).any())
                self.assertFalse((ovr != 0).logical_and(ovr != 1).any())


if __name__ == "__main__":
    CUnitTest.main()
