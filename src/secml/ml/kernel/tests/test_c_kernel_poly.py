from secml.testing import CUnitTest

from secml.array import CArray
from secml.core.type_utils import is_scalar
from secml.data.loader import CDLRandom


class TestCKernelPoly(CUnitTest):
    """Unit test for CKernelPoly."""

    def setUp(self):

        self.d_dense = CDLRandom(n_samples=10, n_features=5,
                                 n_redundant=0, n_informative=3,
                                 n_clusters_per_class=1).load()
        self.d_sparse = self.d_dense.tosparse()

        self.m1 = CArray.rand(shape=(10, 10000), sparse=True, density=0.05)
        self.m2 = CArray.rand(shape=(10, 10000), sparse=True, density=0.05)

        from secml.ml.kernel.c_kernel_poly import CKernelPoly
        self.kernel = CKernelPoly()

    def test_similarity_shape(self):
        """Test shape of kernel."""
        self.logger.info("Testing shape of POLY kernel output.")

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

        cmp_kernel(self.kernel.k, x_vect, x_vect)
        cmp_kernel(self.kernel.k, x_mat, x_vect)
        cmp_kernel(self.kernel.k, x_vect, x_mat)
        cmp_kernel(self.kernel.k, x_mat, x_mat)
        cmp_kernel(self.kernel.k, x_col, x_col)
        cmp_kernel(self.kernel.k, x_col, x_single)
        cmp_kernel(self.kernel.k, x_single, x_col)
        cmp_kernel(self.kernel.k, x_single, x_single)


if __name__ == '__main__':
    CUnitTest.main()
