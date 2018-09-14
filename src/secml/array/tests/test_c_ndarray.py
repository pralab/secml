"""
Created on 27/apr/2015

This module tests the c_ndarray class
If you find any BUG, please notify authors first.

@author: Davide

"""
import unittest
import numpy as np

from secml.utils import CUnitTest
from secml.array import Cdense


class TestCndarray(CUnitTest):
    """Unit test for Cdense."""

    def test_save_load(self):

        self.logger.info("UNITTEST - Cdense - save/load matrix")

        # Cleaning temp file
        try:
            import os
            os.remove('test.txt')
        except (OSError, IOError) as e:
            self.logger.info(e.message)

        a = Cdense().zeros((1000, 1000))

        with self.timer():
            a.save('test.txt')

        with self.timer():
            b = Cdense().load('test.txt', startrow=100, cols=Cdense(np.arange(0, 100)))

        self.assertFalse((a[100:, 0:100] != b).any())

        self.logger.info("UNITTEST - Cdense - save/load vector")

        a = Cdense().zeros(1000, dtype=int)

        with self.timer():
            a.save('test.txt', overwrite=True)

        with self.timer():
            b = Cdense().load('test.txt', cols=slice(100, None, None), dtype=int).ravel()

        self.assertFalse((a[0, 100] != b).any())

        with self.assertRaises(IndexError) as e:
            Cdense().load('test.txt', startrow=10, cols=slice(100, None, None))

        self.logger.info("UNITTEST - Cdense - save/load row vector")

        a = Cdense().zeros((1, 1000))

        with self.timer():
            a.save('test.txt', overwrite=True)

        with self.timer():
            b = Cdense().load('test.txt', cols=Cdense.arange(100, 1000))

        self.assertFalse((a[:, 100:] != b).any())

        with self.assertRaises(IndexError) as e:
            Cdense().load('test.txt', startrow=10, cols=Cdense([3, 4]))
        self.logger.info(e.exception)

        self.logger.info("UNITTEST - Cdense - save/load negative vector")

        a = -Cdense().zeros(1000)

        a.save('test.txt', overwrite=True)
        with open('test.txt', mode='a+') as fhandle:
            with self.timer():
                a.save(fhandle, overwrite=True)

        b = Cdense().load('test.txt')
        # Simulating double save \w append
        a = a.atleast_2d().append(a.atleast_2d(), axis=0)

        self.assertFalse((a != b).any())

        a = Cdense(['a', 'b'])

        with self.timer():
            a.save('test.txt', overwrite=True)

        b = Cdense().load('test.txt', dtype=str).ravel()

        self.assertFalse((a != b).any())

        # Cleaning temp file
        try:
            import os
            os.remove('test.txt')
        except (OSError, IOError) as e:
            self.logger.info(e.message)

       
if __name__ == '__main__':
    unittest.main()
