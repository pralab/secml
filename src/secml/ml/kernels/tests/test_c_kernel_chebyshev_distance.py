from secml.ml.kernels.tests import CCKernelTestCases


class TestCKernelChebyshevDistance(CCKernelTestCases):
    """Unit test for CKernelChebyshevDistance."""

    def setUp(self):
        self._set_up('chebyshev-dist')

    def test_similarity_shape(self):
        """Test shape of kernel."""
        self._test_similarity_shape()
        try:
            self._test_similarity_shape_sparse()
        except TypeError:
            # computation of kernel is not supported on sparse matrices
            pass

    def test_gradient(self):
        self._test_gradient()
        try:
            self._test_gradient_sparse()
        except TypeError:
            # computation of kernel is not supported on sparse matrices
            pass
        self._test_gradient_multiple_points()
        try:
            self._test_gradient_multiple_points_sparse()
        except TypeError:
            # computation of kernel is not supported on sparse matrices
            pass
        self._test_gradient_w()


if __name__ == '__main__':
    CCKernelTestCases.main()
