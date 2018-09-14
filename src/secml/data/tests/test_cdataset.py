"""
Created on 27/apr/2015

This module tests the CDataset class.

@author: Davide Maiorca

If you find any BUG, please notify authors first.

"""
import unittest
from secml.array import CArray
from secml.data import CDataset
from secml.utils import CUnitTest


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
        self.assertEquals(self.dataset.num_classes, 2, "Wrong number of classes!")
        self.assertEquals(self.dataset.num_samples, 3, "Wrong number of patterns!")
        self.assertEquals(self.dataset.num_features, 3, "Wrong number of features!")

    def test_getters_and_setters(self):
        """Test for getters and setters of the class."""
        self.logger.info("Testing setters and getters for the dataset...")
        self.assertEquals((self.dataset.X == self.X).all(), True, "Wrong pattern extraction")
        self.assertEquals((self.dataset.Y == self.Y).all(), True, "Wrong labels extraction")
        new_patterns = CArray([[1, 2, 3], [4, 5, 6]])
        self.logger.info("Setting new patterns: \n" + str(new_patterns))
        self.dataset.X = new_patterns
        self.logger.info("Testing new patterns...")
        self.assertEquals((self.dataset.X == new_patterns).all(), True, "Wrong patterns set!")
        new_labels = CArray([1, 2])
        self.logger.info("Setting new labels: \n" + str(new_labels))
        self.dataset.Y = new_labels
        self.logger.info("Testing new labels...")
        self.assertEquals((self.dataset.Y == new_labels).all(), True, "Wrong labels extraction")

    def test_select_patterns(self):
        """Tests for select patterns method."""
        self.logger.info("Testing pattern extraction...")
        patterns = self.dataset.X[0:2, :]
        target = CArray([[1, 2, 3], [4, 5, 6]])
        self.logger.info("Extracting patterns: \n" + str(patterns) + "\n" + "Targets: \n" + str(target))
        self.logger.info("Testing row extraction...")
        self.assertEquals((patterns == target).all(), True, "Wrong subset extraction")

    def test_subset(self):
        """Tests for subset method."""
        self.logger.info("Testing subsets...")
        subset_lists = [([0, 1], [0, 1]), ([0, 2], slice(0, 3)), (slice(0, 3), [0, 2])]
        x_targets = [CArray([[1, 2], [4, 5]]), CArray([[1, 2, 3], [7, 8, 9]]), CArray([[1, 3], [4, 6], [7, 9]])]
        y_targets = [CArray([1, 2]), CArray([1, 2]), CArray([1, 2, 2])]
        for row_cols, Xtarget, Ytarget in zip(subset_lists, x_targets, y_targets):
            rows = row_cols[0]
            cols = row_cols[1]
            subset = self.dataset[rows, cols]
            self.logger.info(
                "Testing Subset extraction with rows indices: " + str(rows) +
                " and columns indices: " + str(cols) + " \n" + str(subset.X) + " \n" + str(subset.Y))
            self.assertEquals((subset.X == Xtarget).all(), True, "Wrong subset extraction")
            self.assertEquals((subset.Y == Ytarget).all(), True, "Wrong subset extraction")

    def test_custom_attr(self):
        """Testing for custom attributes."""
        ds = CDataset(self.X, self.Y, id='mydataset', age=34, color='green')
        ds_params = ds.get_params()
        self.assertEqual(ds_params['id'], 'mydataset')
        self.assertEqual(ds_params['age'], 34)
        self.assertEqual(ds_params['color'], 'green')


if __name__ == "__main__":
    unittest.main()
