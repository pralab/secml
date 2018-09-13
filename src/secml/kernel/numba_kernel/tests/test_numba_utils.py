import unittest
from prlib.utils import CUnitTest

from prlib.array import CArray
from prlib.kernel.numba_kernel.numba_utils import *


class TestNumbaUtils(CUnitTest):
    """Unit test for Numba utilities collection."""

    def setUp(self):
        # Creating different arrays to use for the tests

        self.v1_dense = CArray([1, 0, 2, 4])
        self.v1_sparse = self.v1_dense.tosparse()

        self.v2_dense = CArray([-1, 2, 0, 5])
        self.v2_sparse = self.v2_dense.tosparse()

        self.m1_dense = CArray([[1, 0, 3], [10, 30, 0]])
        self.m1_sparse = self.m1_dense.tosparse()

        self.m2_dense = CArray([[1, -2, 0], [-1, 0, 20]])
        self.m2_sparse = self.m2_dense.tosparse()

    def test_dist(self):

        self.logger.info("Manhattan Distance - Dense")

        with self.timer():
            manh_dist = manh_dense(
                self.v1_dense.atleast_2d().get_data(),
                self.v2_dense.atleast_2d().get_data(), 0, 0)

        self.assertEqual(manh_dist, 7.0)

        self.logger.info("Squared Euclidean Distance - Dense")

        with self.timer():
            sqrd_eucl_dist = sqrd_eucl_dense(
                self.v1_dense.atleast_2d().get_data(),
                self.v2_dense.atleast_2d().get_data(), 0, 0)

        self.assertEqual(sqrd_eucl_dist, 13.0)

    def test_dot(self):

        self.logger.info("Numpy Dot Product - Dense")
        with self.timer():
            dot_numpy = self.v1_dense.atleast_2d().get_data().dot(
                self.v2_dense.atleast_2d().get_data().T)

        self.logger.info("Numba Dot Product - Dense")
        with self.timer():
            dot_numba = dot_dense(self.v1_dense.atleast_2d().get_data(),
                                  self.v2_dense.atleast_2d().get_data(), 0, 0)

        self.assertEqual(dot_numba, dot_numpy.ravel()[0])

        big_v1_sparse = CArray.rand(
            shape=(100, 10000), sparse=True, density=0.05)
        big_v2_sparse = CArray.rand(
            shape=(100, 10000), sparse=True, density=0.05)

        self.logger.info("Scipy Dot Product - Sparse")
        with self.timer():
            dot_scipy = big_v1_sparse.dot(big_v2_sparse.T)
        self.logger.info("Dot Scipy:\n{:}".format(dot_scipy))

        self.logger.info("Numba Dot Product - Sparse")

        with self.timer():
            dot = dot_sparse(big_v1_sparse.tocsr().data,
                             big_v2_sparse.tocsr().data,
                             big_v1_sparse.tocsr().indices,
                             big_v2_sparse.tocsr().indices,
                             big_v1_sparse.tocsr().indptr,
                             big_v2_sparse.tocsr().indptr)
            dot = CArray(dot)[:-1, :].T[:-1, :].T
        self.logger.info("Dot:\n{:}".format(dot))

        self.logger.info("Numba Dot Product - Sparse 2")

        with self.timer():
            dot2 = dot_sparse2(big_v1_sparse.tocsr().data,
                               big_v2_sparse.tocsr().data,
                               big_v1_sparse.tocsr().indices,
                               big_v2_sparse.tocsr().indices,
                               big_v1_sparse.tocsr().indptr,
                               big_v2_sparse.tocsr().indptr)
            dot2 = CArray(dot2)[:-1, :].T[:-1, :].T
        self.logger.info("Dot2:\n{:}".format(dot2))

        self.assertTrue((dot == dot2).all())
        self.assertTrue((dot == dot_scipy).all())


if __name__ == '__main__':
    unittest.main()
