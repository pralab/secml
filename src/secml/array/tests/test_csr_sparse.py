"""
Created on 27/apr/2015

This module tests the csr_sparse class.

If you find any BUG, please notify authors first.

@author: Davide Maiorca

"""
import unittest
import numpy as np

from secml.utils import CUnitTest
from secml.array import Cdense, Csparse


class Testcsr_Csparse(CUnitTest):
    """Unit test for Csparse."""
     
    def setUp(self):
        """Basic set up."""
        self.dense = Cdense([[1, 0, 0, 0, 5],
                             [2, 4, 0, 0, 0],
                             [3, 4, 5, 0, 0]])
        self.dense_vector = Cdense([1, 0, 0, 0, 3])
        self.sparse_matrix = Csparse(self.dense)
        self.sparse_vector = Csparse(self.dense_vector)

    def test_save_load(self):
        """Test save/load of sparse arrays"""
        self.logger.info("UNITTEST - Csparse - save/load")

        # Cleaning temp file
        try:
            import os
            os.remove('test.txt')
        except (OSError, IOError) as e:
            self.logger.info(e.message)

        self.logger.info("UNITTEST - Csparse - Testing save/load for sparse matrix")

        self.sparse_matrix.save('test.txt')

        self.logger.info("Saving again with overwrite=False... IOError should be raised.")
        with self.assertRaises(IOError) as e:
            self.sparse_matrix.save('test.txt')
        self.logger.info(e.exception)

        loaded_sparse_matrix = Csparse.load('test.txt', dtype=int)

        self.assertFalse((loaded_sparse_matrix != self.sparse_matrix).any(),
                         "Saved and loaded arrays (matrices) are not equal!")

        self.logger.info("UNITTEST - Csparse - Testing save/load for sparse vector")

        self.sparse_vector.save('test.txt', overwrite=True)
        loaded_sparse_vector = Csparse.load('test.txt', dtype=int)

        self.assertFalse((loaded_sparse_vector != self.sparse_vector).any(),
                         "Saved and loaded arrays (vectors) are not equal!")

        # Cleaning temp file
        try:
            import os
            os.remove('test.txt')
        except (OSError, IOError) as e:
            self.logger.info(e.message)

    def mixed_tests(self):

        print(self.sparse_matrix[np.ravel(0)[0], np.ravel(0)[0]])
        print(type(self.sparse_matrix[np.ravel(0)[0], np.ravel(0)[0]]))


if __name__ == '__main__':
    unittest.main()
