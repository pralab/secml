from secml.testing import CUnitTest

from secml.utils import pickle_utils, fm
from secml.array import CArray


class TestPickleUtils(CUnitTest):
    """Unittests for `secml.utils.pickle_utils`."""

    def test_save_load(self):

        a = CArray([1, 2, 3])  # Dummy test array

        # Generate a temp file to test
        import tempfile
        tempdir = tempfile.gettempdir()
        tempfile = fm.join(tempdir, 'secml_testpickle')

        tempfile = pickle_utils.save(tempfile, a)

        a_loaded = pickle_utils.load(tempfile)

        self.assert_array_equal(a_loaded, a)


if __name__ == '__main__':
    CUnitTest.main()
