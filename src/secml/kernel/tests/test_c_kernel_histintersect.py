"""
Created on 4/may/2015
Class to test CKernelHistIntersect

@author: Marco Melis

"""
from prlib.core import settings
settings.USE_NUMBA = True

import unittest
from prlib.utils import CUnitTest

from prlib.array import CArray
from prlib.core.type_utils import is_scalar
from prlib.data.loader import CDLRandom
from prlib.kernel import CKernel


class TestCKernelHistIntersect(CUnitTest):
    """Unit test for CKernelHistIntersect.

    NOTE: FOLLOWING TESTS WORKS ONLY IF NUMBA LIBRARY IS AVAILABLE.

    """

    def setUp(self):

        try:
            import numba
        except ImportError:
            raise ImportError(
                "Unittest only available if `Numba` library is installed")
        
        self.d_dense = CDLRandom(n_samples=10, n_features=5,
                                 n_redundant=0, n_informative=3,
                                 n_clusters_per_class=1).load()
        self.d_sparse = self.d_dense.tosparse()

        self.m1 = CArray.rand(shape=(10, 10000), sparse=True, density=0.05)
        self.m2 = CArray.rand(shape=(10, 10000), sparse=True, density=0.05)

        from prlib.kernel.c_kernel_histintersect import CKernelHistIntersect
        self.kernel = CKernelHistIntersect()
        from prlib.kernel.numba_kernel.c_kernel_histintersect_numba import CKernelHistIntersectNumba
        self.kernel_numba = CKernelHistIntersectNumba()

        # System should create a kernel \w numba instance if numba is available
        self.assertEqual(CKernel.create('hist_intersect').__class__.__name__,
                         CKernelHistIntersectNumba().__class__.__name__)

    def test_similarity(self):
        """Tests similarity function."""
        self.logger.info("Testing HIST kernel - Dense")
        with self.timer():
            simil = self.kernel.k(self.d_dense.X)

        # Tests similarity function \w Numba.
        self.logger.info("Testing HIST kernel \w Numba - Dense")
        with self.timer():
            simil_numba = self.kernel_numba.k(self.d_dense.X)

        self.assertLess((simil - simil_numba).norm_2d(), 1e-6,
                        "Standard and numba kernel are not equal!")

        # self.logger.info("Testing HIST kernel - Sparse")
        # with self.timer():
        #     simil_sparse = self.kernel.k(self.d_sparse.X)
        # self.assertEquals(len(simil_sparse.shape), 2, "Bad shape for kernel")
        #
        # self.assertLess((simil - simil_sparse).norm(), 1e-6,,
        #                 "Standard and numba kernel are not equal!")

        self.logger.info("Testing HIST kernel \w Numba - Sparse")
        with self.timer():
            simil_sparse_numba = self.kernel_numba.k(self.d_sparse.X)

        self.assertLess((simil - simil_sparse_numba).norm_2d(), 1e-6,
                        "Standard and numba kernel are not equal!")

    def test_sparser_data(self):

        self.logger.info("PERFORMANCE tests on very SPARSE data")

        # self.logger.info("Testing HIST kernel timing on sparser data")
        # with self.timer():
        #     self.kernel.k(self.m1, self.m2)

        self.logger.info("Testing HIST kernel \w Numba timing on sparser data")
        with self.timer():
            self.kernel_numba.k(self.m1, self.m2)

    def test_similarity_shape(self):
        """Test shape of kernel."""
        self.logger.info("Testing shape of HIST kernel output.")

        x_vect = CArray.rand(shape=(1, 10)).ravel()
        x_mat = CArray.rand(shape=(10, 10))
        x_col = CArray.rand(shape=(10, 1))
        x_single = CArray.rand(shape=(1, 1))

        def cmp_kernel(a1, a2):
            k = self.kernel_numba.k(a1, a2)
            if isinstance(k, CArray):
                self.logger.info("k shape with inputs {:} {:} is: {:}"
                                 "".format(a1.shape, a2.shape, k.shape))
                self.assertEqual(k.shape, (CArray(a1).atleast_2d().shape[0],
                                           CArray(a2).atleast_2d().shape[0]))
            else:
                self.assertTrue(is_scalar(k))

        cmp_kernel(x_vect, x_vect)
        cmp_kernel(x_mat, x_vect)
        cmp_kernel(x_vect, x_mat)
        cmp_kernel(x_mat, x_mat)
        cmp_kernel(x_col, x_col)
        cmp_kernel(x_col, x_single)
        cmp_kernel(x_single, x_col)
        cmp_kernel(x_single, x_single)


if __name__ == '__main__':
    unittest.main()
