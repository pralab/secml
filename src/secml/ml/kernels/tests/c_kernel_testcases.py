from secml.testing import CUnitTest

from secml.array import CArray
from secml.data.loader import CDLRandom
from secml.core.type_utils import is_scalar
from secml.ml.kernels import CKernel
from secml.optim.function import CFunction


class CCKernelTestCases(CUnitTest):
    def _set_up(self, kernel_name):

        self.d_dense = CDLRandom(n_samples=10, n_features=5,
                                 n_redundant=0, n_informative=3,
                                 n_clusters_per_class=1,
                                 random_state=100).load()

        self.p1_dense = self.d_dense.X[0, :]
        self.p2_dense = self.d_dense.X[1, :]

        self.d_sparse = self.d_dense.tosparse()
        self.p1_sparse = self.d_sparse.X[0, :]
        self.p2_sparse = self.d_sparse.X[1, :]

        self.kernel = CKernel.create(kernel_name)

    def _has_gradient(self):
        try:
            self.kernel.rv = self.p1_dense
            self.kernel.gradient(self.p2_dense)
            return True
        except NotImplementedError:
            return False

    def _cmp_kernel(self, k_fun, a1, a2):
        k = k_fun(a1, a2)
        if isinstance(k, CArray):
            self.logger.info("k shape with inputs {:} {:} is: {:}"
                             "".format(a1.shape, a2.shape, k.shape))
            self.assertEqual(k.shape, (CArray(a1).atleast_2d().shape[0],
                                       CArray(a2).atleast_2d().shape[0]))
        else:
            self.assertTrue(is_scalar(k))

    def _test_similarity_shape(self):
        """Test shape of kernel."""
        self.logger.info(
            "Testing shape of " + self.kernel.class_type + " kernel output.")

        x_vect = CArray.rand(shape=(1, 10)).ravel()
        x_mat = CArray.rand(shape=(10, 10))
        x_col = CArray.rand(shape=(10, 1))
        x_single = CArray.rand(shape=(1, 1))

        self._cmp_kernel(self.kernel.k, x_vect, x_vect)
        self._cmp_kernel(self.kernel.k, x_mat, x_vect)
        self._cmp_kernel(self.kernel.k, x_vect, x_mat)
        self._cmp_kernel(self.kernel.k, x_mat, x_mat)
        self._cmp_kernel(self.kernel.k, x_col, x_col)
        self._cmp_kernel(self.kernel.k, x_col, x_single)
        self._cmp_kernel(self.kernel.k, x_single, x_col)
        self._cmp_kernel(self.kernel.k, x_single, x_single)

    def _test_similarity_shape_sparse(self):
        """Test shape of kernel."""
        self.logger.info(
            "Testing shape of " + self.kernel.class_type + " kernel output.")

        x_vect = CArray.rand(shape=(1, 10)).ravel().tosparse()
        x_mat = CArray.rand(shape=(10, 10)).tosparse()
        x_col = CArray.rand(shape=(10, 1)).tosparse()
        x_single = CArray.rand(shape=(1, 1)).tosparse()

        self._cmp_kernel(self.kernel.k, x_vect, x_vect)
        self._cmp_kernel(self.kernel.k, x_mat, x_vect)
        self._cmp_kernel(self.kernel.k, x_vect, x_mat)
        self._cmp_kernel(self.kernel.k, x_mat, x_mat)
        self._cmp_kernel(self.kernel.k, x_col, x_col)
        self._cmp_kernel(self.kernel.k, x_col, x_single)
        self._cmp_kernel(self.kernel.k, x_single, x_col)
        self._cmp_kernel(self.kernel.k, x_single, x_single)

    def _test_gradient(self):
        """Test for kernel gradients with dense points."""

        if not self._has_gradient():
            self.logger.info(
                "Gradient is not implemented for %s. "
                "Skipping gradient dense tests.", self.kernel.class_type)
            return

        # we invert the order of input patterns as we compute the kernel
        # gradient wrt the second point but check_grad needs it as first input
        def kern_f_for_test(p2, p1, kernel_func):
            return kernel_func.k(p2, p1)

        def kern_grad_for_test(p2, p1, kernel_func):
            kernel_func.rv = p1
            return kernel_func.gradient(p2)

        self.logger.info("Testing gradient with dense data.")
        self.logger.info("Kernel type: %s", self.kernel.class_type)

        for i in range(self.d_dense.num_samples):
            self.logger.info("x point: " + str(self.p2_dense))
            self.logger.info("y point: " + str(self.d_dense.X[i, :]))

            # TODO: implement centered numerical differences.
            # if analytical gradient is zero, numerical estimation does not
            # work, as it is using one-side estimation. We should use centered
            # numerical differences to gain precision.
            self.kernel.rv = self.d_dense.X[i, :]
            grad = self.kernel.gradient(self.p2_dense)
            if grad.norm() >= 1e-10:
                grad_error = CFunction(
                    kern_f_for_test, kern_grad_for_test).check_grad(
                    self.p2_dense, 1e-8, self.d_dense.X[i, :], self.kernel)
                self.logger.info("Gradient approx. error: {:}"
                                 "".format(grad_error))
                self.assertTrue(grad_error < 1e-4)

    def _test_gradient_sparse(self):
        """Test for kernel gradients with sparse points."""

        if not self._has_gradient():
            self.logger.info(
                "Gradient is not implemented for %s. "
                "Skipping gradient sparse tests.", self.kernel.class_type)
            return

        self.logger.info("Testing gradient with sparse data.")
        self.logger.info("Kernel type: %s", self.kernel.class_type)

        self.kernel.rv = self.d_sparse.X
        k_grad = self.kernel.gradient(self.p2_dense)
        self.logger.info(
            "sparse/dense ->.isdense: {:}".format(k_grad.isdense))
        self.assertTrue(k_grad.isdense)

        self.kernel.rv = self.d_dense.X
        k_grad = self.kernel.gradient(self.p2_sparse)
        self.logger.info(
            "dense/sparse ->.issparse: {:}".format(k_grad.issparse))
        self.assertTrue(k_grad.issparse)

        self.kernel.rv = self.d_sparse.X
        k_grad = self.kernel.gradient(self.p2_sparse)
        self.logger.info(
            "sparse/sparse ->.issparse: {:}".format(k_grad.issparse))
        self.assertTrue(k_grad.issparse)

    def _test_gradient_multiple_points(self):
        """Test for kernel gradients with multiple points vs single point."""

        if not self._has_gradient():
            self.logger.info(
                "Gradient is not implemented for %s. "
                "Skipping multiple-point tests.", self.kernel.class_type)
            return

        # check if gradient computed on multiple points is the same as
        # the gradients computed on one point at a time.
        data = self.d_dense.X[0:5, :]  # using same no. of points and features
        self.kernel.rv = data
        k1 = self.kernel.gradient(self.p2_dense)
        k2 = CArray.zeros(shape=k1.shape)
        for i in range(k2.shape[0]):
            self.kernel.rv = data[i, :]
            k2[i, :] = self.kernel.gradient(self.p2_dense)
        self.assertTrue((k1 - k2).ravel().norm() < 1e-4)

        data = self.d_dense.X  # using different no. of points/features
        self.kernel.rv = data
        k1 = self.kernel.gradient(self.p2_dense)
        k2 = CArray.zeros(shape=k1.shape)
        for i in range(k2.shape[0]):
            self.kernel.rv = data[i, :]
            k2[i, :] = self.kernel.gradient(self.p2_dense)
        self.assertTrue((k1 - k2).ravel().norm() < 1e-4)

    def _test_gradient_multiple_points_sparse(self):
        """Test for kernel gradients with multiple points vs single point."""

        if not self._has_gradient():
            self.logger.info(
                "Gradient is not implemented for %s. "
                "Skipping multiple-point tests.", self.kernel.class_type)
            return

        # check if gradient computed on multiple points is the same as
        # the gradients computed on one point at a time.
        data = self.d_sparse.X[0:5, :]  # using same no. of points and features
        self.kernel.rv = data
        k1 = self.kernel.gradient(self.p2_dense)
        k2 = CArray.zeros(shape=k1.shape)
        for i in range(k2.shape[0]):
            self.kernel.rv = data[i, :]
            k2[i, :] = self.kernel.gradient(self.p2_dense)
        self.assertTrue((k1 - k2).ravel().norm() < 1e-4)

        data = self.d_sparse.X  # using different no. of points/features
        self.kernel.rv = data
        k1 = self.kernel.gradient(self.p2_dense)
        k2 = CArray.zeros(shape=k1.shape)
        for i in range(k2.shape[0]):
            self.kernel.rv = data[i, :]
            k2[i, :] = self.kernel.gradient(self.p2_dense)
        self.assertTrue((k1 - k2).ravel().norm() < 1e-4)

    def _test_gradient_w(self):
        """Test for backard passing of w in kernel gradients"""

        if not self._has_gradient():
            self.logger.info(
                "Gradient is not implemented for %s. "
                "Skipping multiple-point tests.", self.kernel.class_type)
            return

        # check if the gradient computed when passing w is the same as the
        # gradient computed with w=None and pre-multiplied with w

        # test on single point
        w = CArray.rand(shape=(1,), random_state=0)
        self.kernel.rv = self.p2_dense
        grad_1 = self.kernel.gradient(self.p1_dense, w=w)
        grad_2 = w * (self.kernel.gradient(self.p1_dense))
        grad_2 = grad_2.ravel()
        self.assertTrue(grad_1.is_vector_like)
        self.assertTrue(grad_2.is_vector_like)
        self.assert_array_almost_equal(grad_1, grad_2, decimal=10)

        # test on multiple points
        w = CArray.rand(shape=(5,), random_state=0)
        self.kernel.rv = self.d_dense[:5, :].X
        grad_1 = self.kernel.gradient(self.p1_dense, w=w)
        grad_2 = w.dot(self.kernel.gradient(self.p1_dense)).ravel()
        self.assertTrue(grad_1.is_vector_like)
        self.assertTrue(grad_2.is_vector_like)
        self.assert_array_almost_equal(grad_1, grad_2, decimal=10)


if __name__ == '__main__':
    CUnitTest.main()
