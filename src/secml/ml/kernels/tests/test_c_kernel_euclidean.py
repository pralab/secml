from secml.ml.kernels.tests import CCKernelTestCases


class TestCKernelEuclidean(CCKernelTestCases):
    """Unit test for CKernelEuclidean."""

    def setUp(self):
        self._set_up('euclidean')

    def test_similarity_shape(self):
        """Test shape of kernel."""
        self.kernel.squared = False
        self._test_similarity_shape()
        self._test_similarity_shape_sparse()
        self.kernel.squared = True
        self._test_similarity_shape()
        self._test_similarity_shape_sparse()

    def test_gradient(self):
        self.kernel.squared = False
        self._test_gradient()
        self._test_gradient_sparse()
        self._test_gradient_multiple_points()
        self._test_gradient_multiple_points()
        self._test_gradient_w()
        self.kernel.squared = True
        self._test_gradient()
        self._test_gradient_sparse()
        self._test_gradient_multiple_points()
        self._test_gradient_multiple_points()
        self._test_gradient_w()


if __name__ == '__main__':
    CCKernelTestCases.main()
