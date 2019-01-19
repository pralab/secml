from secml.utils import CUnitTest
from c_array_testcases import CArrayTestCases


class TestCArrayCopy(CArrayTestCases.TestCArray):
    """Unit test for CArray COPY methods."""

    def test_deepcopy(self):
        """Test for CArray.deepcopy() method."""
        self.logger.info("Test for CArray.deepcopy() method")

        def _deepcopy(array):

            self.logger.info("Array:\n{:}".format(array))

            array_deepcopy = array.deepcopy()
            self.logger.info("Array deepcopied:\n{:}".format(
                array_deepcopy.todense()))

            self.assertEquals(array.issparse, array_deepcopy.issparse)
            self.assertEquals(array.isdense, array_deepcopy.isdense)

            # copy method must return a copy of data
            array_deepcopy[:, :] = 9
            self.assertFalse((array_deepcopy == array).any())

        _deepcopy(self.array_sparse)
        _deepcopy(self.array_dense)


if __name__ == '__main__':
    CUnitTest.main()
