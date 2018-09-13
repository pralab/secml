"""
.. module:: KernelHistIntersectNumba
   :synopsis: Histogram Intersection kernel \w Numba optimization

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from numba import guvectorize

from prlib.array import CArray
from prlib.kernel.c_kernel_histintersect import CKernelHistIntersect


class CKernelHistIntersectNumba(CKernelHistIntersect):
    """Histogram Intersection Kernel with Numba optimization.

    Compute the Histogram Intersection kernel between X and Y::

        K(x, y) = sum^n_i ( min(x[i], y[i]) )

    for each pair of rows in X and in Y.

    Attributes
    ----------
    usenumba : True as current kernel has been optimized with Numba.
    cache_size : int, size of the cache used for kernel computation. Default 100.

    Examples
    --------
    >>> from prlib.array import CArray
    >>> from prlib.kernel.numba_kernel.c_kernel_histintersect_numba import CKernelHistIntersectNumba

    >>> print CKernelHistIntersectNumba().k(CArray([[1,2],[3,4]]), CArray([[10,20],[30,40]]))
    CArray([[ 3.  3.]
     [ 7.  7.]])

    >>> print CKernelHistIntersectNumba().k(CArray([[1,2],[3,4]]))
    CArray([[ 3.  3.]
     [ 3.  7.]])

    """
    usenumba = True

    def _k(self, x, y):
        """Compute the histogram intersection kernel between x and y.

        Parameters
        ----------
        x : CArray or array_like
            First array of shape (n_x, n_features).
        y : CArray or array_like
            Second array of shape (n_y, n_features).

        Returns
        -------
        kernel : CArray
            Kernel between x and y. Array of shape (n_x, n_y).

        Notes
        -----
        Histogram Intersection kernel with Numba optimization is only
        available for the following data types:

        .. hlist::

         * `int32`, `int64`
         * `float32`, `float64`

        See Also
        --------
        :meth:`.CKernel.k` : Main computation interface for kernels.

        """
        x_carray = CArray(x)
        y_carray = CArray(y)
        if x_carray.issparse is True or y_carray.issparse is True:
            # return super(self.__class__, self)._k(x_carray, y_carray)
            k = CArray(self._fast_hist_sparse(x_carray.tocsr().data,
                                              y_carray.tocsr().data,
                                              x_carray.tocsr().indices,
                                              y_carray.tocsr().indices,
                                              x_carray.tocsr().indptr,
                                              y_carray.tocsr().indptr))
            return CArray(CArray(k[:-1, :]).T[:-1, :]).T
        else:
            return CArray(self._fast_hist(
                x_carray.tondarray(), y_carray.tondarray()))

    @staticmethod
    @guvectorize(['void(int32[:,:], int32[:,:], float32[:,:])',
                  'void(int64[:,:], int64[:,:], float64[:,:])',
                  'void(float32[:,:], float32[:,:], float32[:,:])',
                  'void(float64[:,:], float64[:,:], float64[:,:])'],
                 '(m,n),(p,n)->(m,p)', nopython=True)
    def _fast_hist(x, y, k):

        for x_row in range(x.shape[0]):
            for y_row in range(y.shape[0]):

                # Computing min(x,y).sum()
                dist = 0.0
                for feat_idx in range(x.shape[1]):
                    dist += min(x[x_row, feat_idx], y[y_row, feat_idx])

                # Computing final kernel for current rows pair
                k[x_row, y_row] = dist

    @staticmethod
    @guvectorize(['void(int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:,:])',
                  'void(int64[:], int64[:], int32[:], int32[:], int32[:], int32[:], int64[:,:])',
                  'void(float32[:], float32[:], int32[:], int32[:], int32[:], int32[:], float32[:,:])',
                  'void(float64[:], float64[:], int32[:], int32[:], int32[:], int32[:], float64[:,:])'],
                 '(n),(m),(n),(m),(p),(q)->(p,q)', nopython=True)
    def _fast_hist_sparse(x_data, y_data, x_indices, y_indices, x_indptr, y_indptr, k):

        for x_row in xrange(x_indptr.size - 1):
            for y_row in xrange(y_indptr.size - 1):

                dot = 0.0
                last_j = y_indptr[y_row] - 1
                for i in xrange(x_indptr[x_row], x_indptr[x_row + 1]):
                    for j in xrange(last_j + 1, y_indptr[y_row + 1]):
                        if x_indices[i] == y_indices[j]:
                            dot += min(x_data[i], y_data[j])
                            last_j = j
                            break
                        elif x_indices[i] < y_indices[j]:
                            break

                k[x_row, y_row] = dot
