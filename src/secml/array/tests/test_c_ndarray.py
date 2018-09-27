import numpy as np

from secml.utils import CUnitTest, fm
from secml.array import CDense


class TestCndarray(CUnitTest):
    """Unit test for CDense."""

    def test_save_load(self):

        self.logger.info("UNITTEST - CDense - save/load matrix")

        test_file = fm.join(fm.abspath(__file__), 'test.txt')

        # Cleaning test file
        try:
            fm.remove_file(test_file)
        except (OSError, IOError) as e:
            self.logger.info(e.message)

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
                test_file, cols=slice(100, None, None), dtype=int).ravel()

        self.assertFalse((a[0, 100] != b).any())

        with self.assertRaises(IndexError) as e:
            CDense().load(test_file, startrow=10, cols=slice(100, None, None))

        self.logger.info("UNITTEST - CDense - save/load row vector")

        a = CDense().zeros((1, 1000))

        with self.timer():
            a.save(test_file, overwrite=True)

        with self.timer():
            b = CDense().load(test_file, cols=CDense.arange(100, 1000))

        self.assertFalse((a[:, 100:] != b).any())

        with self.assertRaises(IndexError) as e:
            CDense().load(test_file, startrow=10, cols=CDense([3, 4]))
        self.logger.info(e.exception)

        self.logger.info("UNITTEST - CDense - save/load negative vector")

        a = -CDense().zeros(1000)

        a.save(test_file, overwrite=True)
        with open(test_file, mode='a+') as fhandle:
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
            self.logger.info(e.message)

       
if __name__ == '__main__':
    CUnitTest.main()
