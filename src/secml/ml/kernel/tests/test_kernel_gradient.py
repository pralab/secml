from secml.utils import CUnitTest

from secml.ml.kernel import *
from secml.array import CArray
from secml.data.loader import CDLRandom
from secml.optimization import COptimizer
from secml.optimization.function import CFunction


class TestKernelGradient(CUnitTest):
    """Unit test to check kernels'gradient function."""

    def setUp(self):

        self.kernel_types = ('linear', 'rbf', 'poly', 'laplacian')
        self.kernels = [
            CKernel.create(kernel_type) for kernel_type in self.kernel_types]

        # Let's check which classes we loaded
        for kernel in self.kernels:
            self.logger.info("Using kernel from class: {:}"
                             "".format(kernel.__class__))

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

    def test_gradient(self):

        # we invert the order of input patterns as we compute the kernel
        # gradient wrt the second point but check_grad needs it as first input
        def kern_f_for_test(p2, p1, kernel_func):
            return kernel_func.similarity(p1, p2)

        def kern_grad_for_test(p2, p1, kernel_func):
            return kernel_func.gradient(p1, p2)

        self.logger.info("Testing Gradient \w Dense Data")

        for kernel in self.kernels:
            self.logger.info("kernel type: %s", kernel.class_type)
            grad_error = COptimizer(
                CFunction(kern_f_for_test, kern_grad_for_test)).check_grad(
                    self.p2_dense, self.p1_dense, kernel)
            self.logger.info("error committed into own grad calc is {:}"
                             "".format(grad_error))
            self.assertTrue(grad_error < 1e-3)

    # TODO:
    # def test_matricial_gradient(self):
    #
    #     self.logger.info("Testing matricial Gradient")


if __name__ == '__main__':
    CUnitTest.main()
