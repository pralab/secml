from secml.core import settings
settings.SECML_USE_NUMBA = True

from secml.utils import CUnitTest
CUnitTest.importskip("numba")

from secml.array import CArray
from secml.core.type_utils import is_scalar
from secml.data.loader import CDLRandom
from secml.kernel import CKernel


class TestCKernelLinearNumba(CUnitTest):
    """Unit test for CKernelLinearNumba.

    NOTE: FOLLOWING TESTS WORKS ONLY IF NUMBA LIBRARY IS AVAILABLE.

    """

    def setUp(self):

        self.d_dense = CDLRandom(n_samples=10, n_features=500,
                                 n_redundant=0, n_informative=3,
                                 n_clusters_per_class=1).load()
        self.d_sparse = self.d_dense.tosparse()

        self.m1 = CArray.rand(shape=(10, 10000), sparse=True, density=0.05)
        self.m2 = CArray.rand(shape=(10, 10000), sparse=True, density=0.05)

        from secml.kernel.c_kernel_linear import CKernelLinear
        self.kernel = CKernelLinear()
        from secml.kernel.numba_kernel.c_kernel_linear_numba import CKernelLinearNumba
        self.kernel_numba = CKernelLinearNumba()

        # System should create a kernel \w numba instance if numba is available
        self.assertEqual(CKernel.create('linear').__class__.__name__,
                         CKernelLinearNumba().__class__.__name__)

    def test_similarity(self):
        """Tests similarity function."""
        self.logger.info("Testing LINEAR kernel - Dense")
        with self.timer():
            simil = self.kernel.k(self.d_dense.X)

        self.logger.info("Testing LINEAR kernel \w Numba - Dense")
        with self.timer():
            simil_numba = self.kernel_numba.k(self.d_dense.X)

        self.assertLess((simil - simil_numba).norm_2d(), 1e-6,
                        "Standard and numba kernel are not equal!")

        self.logger.info("Testing LINEAR kernel - Sparse")
        with self.timer():
            simil_sparse = self.kernel.k(self.d_sparse.X)

        self.assertLess((simil - simil_sparse).norm_2d(), 1e-6,
                        "Standard and numba kernel are not equal!")

        self.logger.info("Testing LINEAR kernel \w Numba - Sparse")
        with self.timer():
            simil_sparse_numba = self.kernel_numba.k(self.d_sparse.X)

        self.assertLess((simil - simil_sparse_numba).norm_2d(), 1e-6,
                        "Standard and numba kernel are not equal!")

    def test_sparser_data(self):

        self.logger.info("PERFORMANCE tests on SPARSE data")

        self.logger.info("Testing LINEAR kernel time on sparser data")
        with self.timer():
            self.kernel.k(self.m1, self.m2)

        self.logger.info("Testing LINEAR kernel \w Numba time on sparser data")
        with self.timer():
            self.kernel_numba.k(self.m1, self.m2)

    def test_gradient(self):

        """Tests gradient function."""
        self.logger.info("Testing LINEAR gradient - Dense")
        with self.timer():
            gradient = self.kernel.gradient(
                self.d_dense.X, CArray.ones((1, self.d_dense.X.shape[1])))
        self.assertEquals(len(gradient.shape), 2)

        # Tests gradient function \w Numba.
        self.logger.info("Testing LINEAR gradient \w Numba - Dense")
        with self.timer():
            gradient_numba = self.kernel_numba.gradient(
                self.d_dense.X, CArray.ones((1, self.d_dense.X.shape[1])))
        self.assertEquals(len(gradient_numba.shape), 2)

        self.assertLess((gradient - gradient_numba).norm_2d(), 1e-6,
                        "Standard and numba kernel gradient are not equal!")

        self.logger.info("Testing LINEAR gradient - Sparse")
        with self.timer():
            gradient_sparse = self.kernel.gradient(
                self.d_sparse.X, CArray.ones(
                    (1, self.d_sparse.X.shape[1]), sparse=True))
        self.assertEquals(len(gradient_sparse.shape), 2)

        self.assertLess((gradient - gradient_sparse).norm_2d(), 1e-6,
                        "Standard and numba kernel gradient are not equal!")

        self.logger.info("Testing LINEAR gradient \w Numba - Sparse")
        with self.timer():
            gradient_numba_sparse = self.kernel_numba.gradient(
                self.d_sparse.X, CArray.ones(
                    (1, self.d_sparse.X.shape[1]), sparse=True))
        self.assertEquals(len(gradient_numba_sparse.shape), 2)

        self.assertLess((gradient - gradient_numba_sparse).norm_2d(), 1e-6,
                        "Standard and numba kernel gradient are not equal!")

    def test_similarity_shape(self):
        """Test shape of kernel."""
        self.logger.info("Testing shape of LINEAR kernel output.")

        x_vect = CArray.rand(shape=(1, 10)).ravel()
        x_mat = CArray.rand(shape=(10, 10))
        x_col = CArray.rand(shape=(10, 1))
        x_single = CArray.rand(shape=(1, 1))

        def cmp_kernel(k_fun, a1, a2):
            k = k_fun(a1, a2)
            if isinstance(k, CArray):
                self.logger.info("k shape with inputs {:} {:} is: {:}"
                                 "".format(a1.shape, a2.shape, k.shape))
                self.assertEqual(k.shape, (CArray(a1).atleast_2d().shape[0],
                                           CArray(a2).atleast_2d().shape[0]))
            else:
                self.assertTrue(is_scalar(k))

        cmp_kernel(self.kernel_numba.k, x_vect, x_vect)
        cmp_kernel(self.kernel_numba.k, x_mat, x_vect)
        cmp_kernel(self.kernel_numba.k, x_vect, x_mat)
        cmp_kernel(self.kernel_numba.k, x_mat, x_mat)
        cmp_kernel(self.kernel_numba.k, x_col, x_col)
        cmp_kernel(self.kernel_numba.k, x_col, x_single)
        cmp_kernel(self.kernel_numba.k, x_single, x_col)
        cmp_kernel(self.kernel_numba.k, x_single, x_single)


if __name__ == '__main__':
    CUnitTest.main()
