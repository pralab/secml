from secml.ml.kernels.tests import CCKernelTestCases


class TestCKernelLinear(CCKernelTestCases):
    """Unit test for CKernelLinear."""

    def setUp(self):
        self._set_up('linear')

    def test_similarity_shape(self):
        """Test shape of kernel."""
        self._test_similarity_shape()
        self._test_similarity_shape_sparse()

    def test_gradient(self):
        self._test_gradient()
        self._test_gradient_sparse()
        self._test_gradient_multiple_points()
        self._test_gradient_multiple_points_sparse()
        self._test_gradient_w()


if __name__ == '__main__':
    CCKernelTestCases.main()
