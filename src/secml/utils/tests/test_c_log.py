import unittest
import time

from secml.utils import CLog, CTimer


class TestCLog(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.logger = CLog(logger_id=cls.__name__,
                          add_stream=True,
                          level='DEBUG')

    def test_timed_nologging(self):

        # Calling class CTimer directly
        timer = CTimer()  # Does nothing... Use as context manager!

        # Test for predefined interval
        with timer as t:
            time.sleep(2)
            self.assertGreaterEqual(t.step, 2000)
        self.assertGreaterEqual(t.interval, 2000)

    def test_timed_logging(self):

        from secml.array import CArray

        timer = self.logger.timer()  # Does nothing... Use as context manager!

        # Test for predefined interval
        with timer as t:
            time.sleep(2)
            self.assertGreaterEqual(t.step, 2000)
        self.assertGreaterEqual(t.interval, 2000)

        # Testing logging of method run time
        with self.logger.timer():
            a = CArray.arange(-5, 100, 0.1).transpose()
            a.sort(inplace=True)

        # Test for predefined interval with error
        with self.assertRaises(TypeError):
            with self.logger.timer() as t:
                time.sleep('test')
        self.logger.info("Interval " + str(t.interval) +
                         " should have been logged anyway")


if __name__ == '__main__':
    unittest.main()
