from secml.ml.kernel.tests import CCKernelTestCases
from secml.ml.kernel import CKernelEuclidean


class TestCKernelEuclidean(CCKernelTestCases):
    """Unit test for CKernelHamming."""

    def setUp(self):
        self._set_up('euclidean')

    def test_similarity_shape(self):
        """Test shape of kernel."""
        self._test_similarity_shape()
        self._test_similarity_shape_sparse()

    def test_gradient(self):
        self._test_gradient()
        self._test_gradient_sparse()
        self._test_gradient_multiple_points()

    # TODO test when squared=True. but this needs to be passed to __init__


if __name__ == '__main__':
    CCKernelTestCases.main()
