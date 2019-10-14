from secml.ml.kernel.tests import CCKernelTestCases


class TestCKernelEuclidean(CCKernelTestCases):
    """Unit test for CKernelHamming."""

    def setUp(self):
        self._set_up('euclidean')

    def test_similarity_shape(self):
        """Test shape of kernel."""
        self._test_similarity_shape()

    def test_gradient(self):
        self._test_gradient()
        self._test_gradient_sparse()


if __name__ == '__main__':
    CCKernelTestCases.main()
