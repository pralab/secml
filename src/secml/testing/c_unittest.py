"""
.. module:: UnitTest
   :synopsis: Class for manage unittests

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
import unittest
import pytest
import numpy as np
import numpy.testing as npt

from secml import global_filterwarnings
from secml.utils import CLog
from secml.settings import SECML_STORE_LOGS


class CUnitTest(unittest.TestCase):
    """Superclass for unittests.

    Provides a wrapper of `unittests.TestCase` in addition to:

     - integrated logging functionalities (see `CLog`)
     - integrated timing functionalities (see `CTimer`)
     - addition assertion methods from `numpy.testing`
     - `skip`, `skipif`, `importorskip` functions from `pytest`

    """

    @property
    def logger(self):
        """Logger for current object."""
        return self._logger.get_child(self.__class__.__name__)

    @classmethod
    def setUpClass(cls):
        # Unittests logs will be stored in a `unittests.log` file
        # in the same directory of the calling unittest.
        # Use `store_logs=True` in configuration file to activate this
        cls._logger = CLog(
            logger_id='unittest', add_stream=True,
            file_handler='unittests.log' if SECML_STORE_LOGS is True else None)
        cls._logger.set_level('DEBUG')

        # As per python 3.2 filterwarnings are reset by unittests for each
        # TestCase. We thus need to restore our filter for every TestCase
        global_filterwarnings()

    def timer(self):
        """Returns a CTimer to be used as context manager."""
        return self.logger.timer()

    # Raises an AssertionError if two array_like objects are not equal
    def assert_array_equal(self, x, y, err_msg='', verbose=True):
        x = x.tondarray() if hasattr(x, 'tondarray') else x
        y = y.tondarray() if hasattr(y, 'tondarray') else y
        return npt.assert_array_equal(x, y, err_msg, verbose)
    assert_array_equal.__doc__ = npt.assert_array_equal.__doc__

    # AssertionError if two objects are not equal up to desired precision
    def assert_array_almost_equal(
            self, x, y, decimal=6, err_msg='', verbose=True):
        x = x.tondarray() if hasattr(x, 'tondarray') else x
        y = y.tondarray() if hasattr(y, 'tondarray') else y
        return npt.assert_array_almost_equal(x, y, decimal, err_msg, verbose)
    assert_array_almost_equal.__doc__ = npt.assert_array_almost_equal.__doc__

    # Compare two arrays relatively to their spacing
    def assert_array_almost_equal_nulp(self, x, y, nulp=1):
        x = x.tondarray() if hasattr(x, 'tondarray') else x
        y = y.tondarray() if hasattr(y, 'tondarray') else y
        return npt.assert_array_almost_equal_nulp(x, y, nulp)
    assert_array_almost_equal_nulp.__doc__ = npt.assert_array_almost_equal_nulp.__doc__

    # AssertionError if two array_like objects are not ordered by less than
    def assert_array_less(self, x, y, err_msg='', verbose=True):
        x = x.tondarray() if hasattr(x, 'tondarray') else x
        y = y.tondarray() if hasattr(y, 'tondarray') else y
        return npt.assert_array_less(x, y, err_msg, verbose)
    assert_array_less.__doc__ = npt.assert_array_less.__doc__

    # Check that all elems differ in at most N Units in the last place
    def assert_array_max_ulp(self, a, b, maxulp=1, dtype=None):
        a = a.tondarray() if hasattr(a, 'tondarray') else a
        b = b.tondarray() if hasattr(b, 'tondarray') else b
        return npt.assert_array_max_ulp(a, b, maxulp, dtype)
    assert_array_max_ulp.__doc__ = npt.assert_array_max_ulp.__doc__

    # AssertionError if two objects are not equal up to desired tolerance
    def assert_allclose(self, actual, desired, rtol=1e-6, atol=0,
                        equal_nan=True, err_msg='', verbose=True):
        actual = actual.tondarray() if hasattr(actual, 'tondarray') else actual
        des = desired.tondarray() if hasattr(desired, 'tondarray') else desired
        return npt.assert_allclose(
            actual, des, rtol, atol, equal_nan, err_msg, verbose)
    assert_allclose.__doc__ = npt.assert_allclose.__doc__

    # AssertionError if two items are not equal up to significant digits.
    def assert_approx_equal(
            self, actual, desired, significant=6, err_msg='', verbose=True,):
        actual = actual.tondarray() if hasattr(actual, 'tondarray') else actual
        des = desired.tondarray() if hasattr(desired, 'tondarray') else desired
        return npt.assert_approx_equal(
            actual, des, significant, err_msg, verbose)
    assert_approx_equal.__doc__ = npt.assert_approx_equal.__doc__

    @staticmethod
    def assertIsSubDtype(actual, desired):
        """
        Raises an AssertionError if the first dtype is NOT a typecode
        lower/equal in type hierarchy.

        Parameters
        ----------
        actual : dtype or str
            dtype or string representing a typecode.
        desired : dtype or str
            dtype or string representing a typecode.

        """
        if desired is float:
            # To manage the built-in float as either np.float32 or np.float64
            desired = np.floating
        elif desired is int:
            # To manage the built-in int as either np.int32 or np.int64
            desired = np.integer
        else:  # Convert built-in types to numpy dtypes for using issubdtype
            desired = np.dtype(desired).type
        if not np.issubdtype(actual, desired):
            raise AssertionError("{:} is not lower/equal to {:} in the type "
                                 "hierarchy.".format(actual, desired))

    # Skip something
    skip = pytest.mark.skip

    # Skip something conditionally
    skipif = pytest.mark.skipif

    # Skipping on a missing import dependency
    importskip = staticmethod(pytest.importorskip)

    # Wrapper to use "if __name__ == '__main__': CUnitTest.main()"
    main = unittest.main
