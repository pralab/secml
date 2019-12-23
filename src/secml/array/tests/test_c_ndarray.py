from secml.testing import CUnitTest

import numpy as np

from secml.utils import fm
from secml.array.c_dense import CDense


class TestCDense(CUnitTest):
    """Unit test for CDense."""

    def test_save_load(self):

        self.logger.info("UNITTEST - CDense - save/load matrix")

        test_file = fm.join(fm.abspath(__file__), 'test.txt')

        # Cleaning test file
        try:
            fm.remove_file(test_file)
        except (OSError, IOError) as e:
            if e.errno != 2:
                raise e

        a = CDense().zeros((1000, 1000))

        with self.timer():
            a.save(test_file)

        with self.timer():
            b = CDense().load(
                test_file, startrow=100, cols=CDense(np.arange(0, 100)))

        self.assertFalse((a[100:, 0:100] != b).any())

        self.logger.info("UNITTEST - CDense - save/load vector")

        a = CDense().zeros(1000, dtype=int)

        with self.timer():
            a.save(test_file, overwrite=True)

        with self.timer():
            b = CDense().load(
                test_file, cols=list(range(100, 1000)), dtype=int).ravel()

        self.assertFalse((a[0, 100] != b).any())

        if np.__version__ < '1.18':
            with self.assertRaises(IndexError) as e:
                CDense().load(test_file, startrow=10)
            self.logger.info("Expected error: {:}".format(e.exception))
        else:
            with self.logger.catch_warnings():
                self.logger.filterwarnings(
                    "ignore", message="genfromtxt: Empty input file")
                a = CDense().load(test_file, startrow=10)
                self.assertEqual(a.size, 0)

        self.logger.info("UNITTEST - CDense - save/load row vector")

        a = CDense().zeros((1, 1000))

        with self.timer():
            a.save(test_file, overwrite=True)

        with self.timer():
            b = CDense().load(test_file, cols=CDense.arange(100, 1000))

        self.assertFalse((a[:, 100:] != b).any())

        # For some reasons np.genfromtxt does not close the file here
        # Let's handle the resource warning about unclosed file
        with self.logger.catch_warnings():
            self.logger.filterwarnings("ignore", message="unclosed file")
            if np.__version__ < '1.18':
                with self.assertRaises(IndexError) as e:
                    CDense().load(test_file, startrow=10)
                    self.logger.info("Expected error: {:}".format(e.exception))
            else:
                self.logger.filterwarnings(
                    "ignore", message="genfromtxt: Empty input file")
                a = CDense().load(test_file, startrow=10)
                self.assertEqual(a.size, 0)

        self.logger.info("UNITTEST - CDense - save/load negative vector")

        a = -CDense().zeros(1000)

        a.save(test_file, overwrite=True)
        with open(test_file, mode='at+') as fhandle:
            with self.timer():
                a.save(fhandle, overwrite=True)

        b = CDense().load(test_file)
        # Simulating double save \w append
        a = a.atleast_2d().append(a.atleast_2d(), axis=0)

        self.assertFalse((a != b).any())

        a = CDense(['a', 'b'])

        with self.timer():
            a.save(test_file, overwrite=True)

        b = CDense().load(test_file, dtype=str).ravel()

        self.assertFalse((a != b).any())

        # Cleaning test file
        try:
            fm.remove_file(test_file)
        except (OSError, IOError) as e:
            if e.errno != 2:
                raise e

       
if __name__ == '__main__':
    CUnitTest.main()
