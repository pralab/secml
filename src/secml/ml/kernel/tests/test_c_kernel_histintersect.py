from secml.ml.kernel.tests import CCKernelTestCases


class TestCKernelHistIntersect(CCKernelTestCases):
    """Unit test for CKernelHistIntersect."""

    def setUp(self):
        self._set_up('hist-intersect')

    def test_similarity_shape(self):
        """Test shape of kernel."""
        self._test_similarity_shape()


if __name__ == '__main__':
    CCKernelTestCases.main()
