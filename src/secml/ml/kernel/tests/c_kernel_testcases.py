from secml.testing import CUnitTest

from secml.array import CArray
from secml.data.loader import CDLRandom
from secml.core.type_utils import is_scalar
from secml.ml.kernel import CKernel
from secml.optim.function import CFunction


class CCKernelTestCases(CUnitTest):
    def _set_up(self, kernel_name):

        self.d_dense = CDLRandom(n_samples=10, n_features=5,
                                 n_redundant=0, n_informative=3,
                                 n_clusters_per_class=1).load()

        self.p1_dense = self.d_dense.X[0, :]
        self.p2_dense = self.d_dense.X[1, :]

        self.d_sparse = self.d_dense.tosparse()
        self.p1_sparse = self.d_sparse.X[0, :]
        self.p2_sparse = self.d_sparse.X[1, :]

        self.m1 = CArray.rand(shape=(10, 10000), sparse=True, density=0.05)
        self.m2 = CArray.rand(shape=(10, 10000), sparse=True, density=0.05)

        self.kernel = CKernel.create(kernel_name)

    def _has_gradient(self):
        try:
            self.kernel.gradient(self.p1_dense, self.p2_dense)
            return True
        except NotImplementedError:
            return False

    def _test_similarity_shape(self):
        """Test shape of kernel."""
        self.logger.info(
            "Testing shape of " + self.kernel.class_type + " kernel output.")

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

    def _test_gradient(self):

        if not self._has_gradient():
            self.logger.info(
                "Gradient is not implemented for %s. "
                "Skipping gradient dense tests.", self.kernel.class_type)
            return

        # we invert the order of input patterns as we compute the kernel
        # gradient wrt the second point but check_grad needs it as first input
        def kern_f_for_test(p2, p1, kernel_func):
            return kernel_func.similarity(p1, p2)

        def kern_grad_for_test(p2, p1, kernel_func):
            return kernel_func.gradient(p1, p2)

        self.logger.info("Testing gradient with dense data.")
        self.logger.info("Kernel type: %s", self.kernel.class_type)

        grad_error = CFunction(
            kern_f_for_test, kern_grad_for_test).check_grad(
            self.p2_dense, 1e-8, self.p1_dense, self.kernel)

        self.logger.info("Gradient approximation: {:}"
                         "".format(grad_error))

        self.assertTrue(grad_error < 1e-3)

    def _test_gradient_sparse(self):
        """Test for kernel gradients with sparse points."""

        if not self._has_gradient():
            self.logger.info(
                "Gradient is not implemented for %s. "
                "Skipping gradient sparse tests.", self.kernel.class_type)
            return

        self.logger.info("Testing gradient with sparse data.")
        self.logger.info("Kernel type: %s", self.kernel.class_type)

        k_grad = self.kernel.gradient(self.p1_sparse, self.p2_dense)
        self.logger.info(
            "sparse/dense ->.isdense: {:}".format(k_grad.isdense))
        self.assertTrue(k_grad.isdense)

        k_grad = self.kernel.gradient(self.p1_dense, self.p2_sparse)
        self.logger.info(
            "dense/sparse ->.issparse: {:}".format(k_grad.issparse))
        self.assertTrue(k_grad.issparse)

        k_grad = self.kernel.gradient(self.p1_sparse, self.p2_sparse)
        self.logger.info(
            "sparse/sparse ->.issparse: {:}".format(k_grad.issparse))
        self.assertTrue(k_grad.issparse)


if __name__ == '__main__':
    CUnitTest.main()
