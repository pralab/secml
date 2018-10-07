"""
.. module:: UnitTest
   :synopsis: Class for manage unittests

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
import unittest
import pytest
from secml.utils import CLog


class CUnitTest(unittest.TestCase):
    """Superclass for unittests.

    .. warning::
        Keep this class inheriting only from the unittest platform.
        The testing environment should be as clean as possible.

    """

    @property
    def logger(self):
        """Logger for current object."""
        return self._logger.get_child(self.__class__.__name__)

    @classmethod
    def setUpClass(cls):
        # TODO: MAKE FILE PATH/NAME DYNAMIC
        cls._logger = CLog(logger_id='unittest', add_stream=True,
                           file_handler='unittest.log')
        cls._logger.set_level('DEBUG')

    def timer(self):
        """Returns a CTimer to be used as context manager."""
        return self.logger.timer()

    # Skip something
    skip = pytest.mark.skip

    # Skip something conditionally
    skipif = pytest.mark.skipif

    # Skipping on a missing import dependency
    importskip = staticmethod(pytest.importorskip)

    # Wrapper to use "if __name__ == '__main__': CUnitTest.main()"
    main = unittest.main
