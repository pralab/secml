"""
.. module:: KernelLinearNumba
   :synopsis: Linear kernel \w Numba optimization

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from numba import guvectorize

from secml.array import CArray
from secml.kernel.c_kernel_linear import CKernelLinear

from secml.kernel.numba_kernel.numba_utils import sqrd_eucl_dense, dot_dense


class CKernelLinearNumba(CKernelLinear):
    """Linear kernel with Numba optimization.

    Compute the Linear kernel between X and Y::

        K(x, y) = x * y^T

    for each pair of rows in X and in Y.

    Attributes
    ----------
    usenumba : True as current kernel has been optimized with Numba.
    cache_size : int, size of the cache used for kernel computation. Default 100.

    Examples
    --------
    >>> from secml.array import CArray
    >>> from secml.kernel.numba_kernel.c_kernel_linear_numba import CKernelLinearNumba

    >>> print CKernelLinearNumba().k(CArray([[1,2],[3,4]]), CArray([[10,20],[30,40]]))
    CArray([[  50.  110.]
     [ 110.  250.]])

    >>> print CKernelLinearNumba().k(CArray([[1,2],[3,4]]))
    CArray([[  5.  11.]
     [ 11.  25.]])

    """
    usenumba = True

    def _k(self, x, y):
        """Compute the Linear kernel between x and y.

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
        Linear kernel with Numba optimization is only available for
        the following data types:

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
            return super(self.__class__, self)._k(x_carray, y_carray)
        else:
            return CArray(self._fast_linear(
                x_carray.tondarray(), y_carray.tondarray()))

    def gradient(self, x, v):
        """Calculates Linear kernel gradient wrt vector 'v'.

        The gradient of Linear kernel is given by::

            dK(x[i],v)/dv =     x[i]  if x[i] != v
                          = 2 * x[i]  if x[i] == v

        where x[i] is the i-th sample of first input array.

        Parameters
        ----------
        x : CArray or array_like
            First array of shape (n_x, n_features).
        v : CArray or array_like
            Second array of shape (n_features, ) or (1, n_features).

        Returns
        -------
        kernel_gradient : CArray
            Kernel gradient of x with respect to vector v. Array of
            shape (n_x, n_features) if n_x > 1, else a flattened
            array of shape (n_features, ).

        Notes
        -----
        Gradient of Linear kernel with Numba optimization
        is only available for the following data types:

        .. hlist::

         * `int32`, `int64`
         * `float32`, `float64`

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.kernel.numba_kernel.c_kernel_linear_numba import CKernelLinearNumba

        >>> array = CArray([[15,25],[45,55]])
        >>> vector = CArray([2,5])
        >>> print CKernelLinearNumba().gradient(array, vector)
        CArray([[ 15.  25.]
         [ 45.  55.]])

        >>> print CKernelLinearNumba().gradient(vector, vector)
        CArray([  4.  10.])

        """
        x_carray = CArray(x)
        v_carray = CArray(v)
        if x_carray.issparse is True or v_carray.issparse is True:
            return super(self.__class__, self).gradient(x, v)
        else:
            k_grad = self._gradient(
                x_carray.atleast_2d(), v_carray.atleast_2d())
            return k_grad if k_grad.shape[0] > 1 else k_grad.ravel()

    def _gradient(self, x, v):
        """Numba function to compute Linear kernel gradient wrt vector 'v'.

        Parameters
        ----------
        x : CArray or array_like
            First array of shape (n_x, n_features).
        v : CArray or array_like
            Second array of shape (1, n_features).

        Returns
        -------
        kernel_gradient : CArray
            Kernel gradient of x with respect to vector v. Array of
            shape (n_x, n_features).

        See Also
        --------
        :meth:`.CKernelLinearNumba.gradient` : Gradient of Linear kernel with Numba optimization.

        """
        x_carray = CArray(x)
        v_carray = CArray(v)
        if x_carray.issparse is True or v_carray.issparse is True:
            return super(self.__class__, self)._gradient(x, v)
        else:
            return CArray(self._fast_linear_gradient(
                x_carray.tondarray(), v_carray.tondarray()))

    @staticmethod
    @guvectorize(['void(int32[:,:], int32[:,:], float32[:,:])',
                  'void(int64[:,:], int64[:,:], float64[:,:])',
                  'void(float32[:,:], float32[:,:], float32[:,:])',
                  'void(float64[:,:], float64[:,:], float64[:,:])'],
                 '(m,n),(p,n)->(m,p)', nopython=True)
    def _fast_linear(x, y, k):

        for x_row in range(x.shape[0]):
            for y_row in range(y.shape[0]):

                # Computing x^T * y and final kernel for current rows pair
                k[x_row, y_row] = dot_dense(x, y, x_row, y_row)

    @staticmethod
    @guvectorize(['void(int32[:,:], int32[:,:], float32[:,:])',
                  'void(int64[:,:], int64[:,:], float64[:,:])',
                  'void(float32[:,:], float32[:,:], float32[:,:])',
                  'void(float64[:,:], float64[:,:], float64[:,:])'],
                 '(m,n),(p,n)->(m,n)', nopython=True)
    def _fast_linear_gradient(x, v, k_grad):

        for x_row in range(x.shape[0]):

            # Check if current x row and v are the same vector
            if sqrd_eucl_dense(x, v, x_row, 0) < 1e-8:
                # if x row and v are the same, return 2 * x
                factor = 2.0
            else:
                factor = 1.0

            # Computing final kernel
            for feat_idx in range(x.shape[1]):
                k_grad[x_row, feat_idx] = factor * x[x_row, feat_idx]
