"""
.. module:: NumbaUtils
   :synopsis: Collection of utility functions optimized with Numba library

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from numba import jit, guvectorize
import numpy as np

__all__ = ['manh_dense', 'sqrd_eucl_dense',
           'dot_dense', 'dot_sparse', 'dot_sparse2']


@jit(['float32(int32[:,:], int32[:,:], int32, int32)',
      'float64(int64[:,:], int64[:,:], int64, int64)',
      'float32(float32[:,:], float32[:,:], int32, int32)',
      'float64(float64[:,:], float64[:,:], int64, int64)'], nopython=True)
def manh_dense(x, y, x_row, y_row):
    """Manhattan distance of x[x_row] and y[y_row] \w Numba.

    Given by::

        d = |x[x_row] - y[y_row]| = sum_i abs(x[x_row, i] - y[y_row, i])

    x, y = CArrays
    x_row, y_row = integers

    """
    norm1 = 0.0
    for feat_idx in range(x.shape[1]):
        tmp = x[x_row, feat_idx] - y[y_row, feat_idx]
        norm1 += np.abs(tmp)

    return norm1


@jit(['float32(int32[:,:], int32[:,:], int32, int32)',
      'float64(int64[:,:], int64[:,:], int64, int64)',
      'float32(float32[:,:], float32[:,:], int32, int32)',
      'float64(float64[:,:], float64[:,:], int64, int64)'], nopython=True)
def sqrd_eucl_dense(x, y, x_row, y_row):
    """Squared Euclidean distance of x[x_row] and y[y_row] \w Numba.

    Given by::

        d = ||x[x_row] - y[y_row]||^2 = sum_i (x[x_row, i] - y[y_row, i])^2

    x, y = CArrays
    x_row, y_row = integers

    """
    norm2 = 0.0
    for feat_idx in range(x.shape[1]):
        tmp = x[x_row, feat_idx] - y[y_row, feat_idx]
        norm2 += tmp * tmp

    return norm2


@jit(['float32(int32[:,:], int32[:,:], int32, int32)',
      'float64(int64[:,:], int64[:,:], int64, int64)',
      'float32(float32[:,:], float32[:,:], int32, int32)',
      'float64(float64[:,:], float64[:,:], int64, int64)'], nopython=True)
def dot_dense(x, y, x_row, y_row):
    """Dot Product of x[x_row] and y[y_row] \w Numba.

    Given by::

        dot = x[x_row] * y[y_row]^T

    x, y = CArrays
    x_row, y_row = integers

    """
    dot = 0.0
    for feat_idx in range(x.shape[1]):
        dot += x[x_row, feat_idx] * y[y_row, feat_idx]

    return dot


@guvectorize(['void(int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:,:])',
              'void(int64[:], int64[:], int32[:], int32[:], int32[:], int32[:], int64[:,:])',
              'void(float32[:], float32[:], int32[:], int32[:], int32[:], int32[:], float32[:,:])',
              'void(float64[:], float64[:], int32[:], int32[:], int32[:], int32[:], float64[:,:])'],
             '(n),(m),(n),(m),(p),(q)->(p,q)', nopython=True)
def dot_sparse(x_data, y_data, x_indices, y_indices, x_indptr, y_indptr, k):
    """Dot Product for sparse arrays \w Numba.

    Takes CSparse internal structures as input (indices, indptr, data).

    """
    for x_row in xrange(x_indptr.size - 1):
        for y_row in xrange(y_indptr.size - 1):

            dot = 0.0
            last_j = y_indptr[y_row] - 1
            for i in xrange(x_indptr[x_row], x_indptr[x_row + 1]):
                for j in xrange(last_j + 1, y_indptr[y_row + 1]):
                    if x_indices[i] == y_indices[j]:
                        dot += x_data[i] * y_data[j]
                        last_j = j
                        break
                    elif x_indices[i] < y_indices[j]:
                        break

            k[x_row, y_row] = dot


@guvectorize(['void(int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:,:])',
              'void(int64[:], int64[:], int32[:], int32[:], int32[:], int32[:], int64[:,:])',
              'void(float32[:], float32[:], int32[:], int32[:], int32[:], int32[:], float32[:,:])',
              'void(float64[:], float64[:], int32[:], int32[:], int32[:], int32[:], float64[:,:])'],
             '(n),(m),(n),(m),(p),(q)->(p,q)', nopython=True)
def dot_sparse2(x_data, y_data, x_indices, y_indices, x_indptr, y_indptr, k):
    """Dot Product for sparse arrays \w Numba.

    Takes CSparse internal structures as input (indices, indptr, data).

    """
    last_j = np.empty(y_indptr.size - 1, dtype=np.int32)

    for x_row in xrange(x_indptr.size - 1):
        for i in xrange(x_indptr[x_row], x_indptr[x_row + 1]):
            for y_row in xrange(y_indptr.size - 1):

                if i == x_indptr[x_row]:
                    k[x_row, y_row] = 0
                    last_j[y_row] = y_indptr[y_row] - 1

                for j in xrange(last_j[y_row] + 1, y_indptr[y_row + 1]):

                    if x_indices[i] < y_indices[j]:
                        break
                    elif x_indices[i] == y_indices[j]:
                        k[x_row, y_row] += x_data[i] * y_data[j]
                        last_j[y_row] = j
                        break
