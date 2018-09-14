"""
.. module:: KernelRBFNumba
   :synopsis: Radial basis function (RBF) kernel \w Numba optimization

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from numba import guvectorize
from math import exp

from secml.array import CArray
from secml.kernel.c_kernel_rbf import CKernelRBF

from secml.kernel.numba_kernel.numba_utils import sqrd_eucl_dense


class CKernelRBFNumba(CKernelRBF):
    """Radial basis function (RBF) Kernel with Numba optimization.

    Compute the RBF (gaussian) kernel between X and Y::

        K(x, y) = exp(-gamma ||x-y||^2)

    for each pair of rows in X and in Y.

    Attributes
    ----------
    usenumba : True as current kernel has been optimized with Numba.
    cache_size : int, size of the cache used for kernel computation. Default 100.

    Parameters
    ----------
    gamma : float
        Default is 1.0. Equals to `-0.5*sigma^-2` in the standard
        formulation of rbf kernel, is a free parameter to be used
        for balancing.

    Examples
    --------
    >>> from secml.array import CArray
    >>> from secml.kernel.numba_kernel.c_kernel_rbf_numba import CKernelRBFNumba

    >>> print CKernelRBFNumba(gamma=0.001).k(CArray([[1,2],[3,4]]), CArray([[10,20],[30,40]]))
    CArray([[ 0.66697681  0.10177406]
     [ 0.73712337  0.13199384]])

    >>> print CKernelRBFNumba().k(CArray([[1,2],[3,4]]))
    CArray([[  1.00000000e+00   3.35462628e-04]
     [  3.35462628e-04   1.00000000e+00]])

    """
    usenumba = True

    def _k(self, x, y):
        """Compute the RBF kernel between x and y.

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
        RBF kernel with Numba optimization is only available for
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
            return CArray(self._fast_rbf(
                x_carray.tondarray(), y_carray.tondarray(), self.gamma))

    def gradient(self, x, v):
        """Calculates RBF kernel gradient wrt vector 'v'.

        The gradient of RBF kernel is given by::

            dK(x,v)/dv = 2 * gamma * k(x,v) * (x - v)

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
        Gradient of RBF kernel with Numba optimization is only available for
        the following data types:

        .. hlist::

         * `int32`, `int64`
         * `float32`, `float64`

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.kernel.numba_kernel.c_kernel_rbf_numba import CKernelRBFNumba

        >>> array = CArray([[15,25],[45,55]])
        >>> vector = CArray([2,5])
        >>> print CKernelRBFNumba(gamma=1e-4).gradient(array, vector)
        CArray([[ 0.00245619  0.00377875]
         [ 0.00556703  0.00647329]])

        >>> print CKernelRBFNumba().gradient(vector, vector)
        CArray([ 0.  0.])

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
        """Calculate RBF kernel gradient wrt vector 'v'.

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
            shape (n_x, n_features)

        See Also
        --------
        :meth:`.CKernelRBFNumba.gradient` : Gradient of RBF kernel with Numba optimization.

        """
        x_carray = CArray(x)
        v_carray = CArray(v)
        if x_carray.issparse is True or v_carray.issparse is True:
            return super(self.__class__, self)._gradient(x, v)
        else:
            return CArray(self._fast_rbf_gradient(
                x_carray.tondarray(), v_carray.tondarray(), self.gamma))

    @staticmethod
    @guvectorize(['void(int32[:,:], int32[:,:], float32[:], float32[:,:])',
                  'void(int64[:,:], int64[:,:], float64[:], float64[:,:])',
                  'void(float32[:,:], float32[:,:], float32[:], float32[:,:])',
                  'void(float64[:,:], float64[:,:], float64[:], float64[:,:])'],
                 '(m,n),(p,n),()->(m,p)', nopython=True)
    def _fast_rbf(x, y, gamma, k):

        for x_row in range(x.shape[0]):
            for y_row in range(y.shape[0]):

                # Computing ||x - v||^2 and final kernel for current rows pair
                k[x_row, y_row] = exp(-gamma[0] * sqrd_eucl_dense(x, y, x_row, y_row))

    @staticmethod
    @guvectorize(['void(int32[:,:], int32[:,:], float32[:], float32[:,:])',
                  'void(int64[:,:], int64[:,:], float64[:], float64[:,:])',
                  'void(float32[:,:], float32[:,:], float32[:], float32[:,:])',
                  'void(float64[:,:], float64[:,:], float64[:], float64[:,:])'],
                 '(m,n),(p,n),()->(m,n)', nopython=True)
    def _fast_rbf_gradient(x, v, gamma, k_grad):

        for x_row in range(x.shape[0]):

            # Computing ||x - v||^2 and final kernel for current row
            k = exp(-gamma[0] * sqrd_eucl_dense(x, v, x_row, 0))

            # Computing 2 * gamma * k * (x - v)
            for feat_idx in range(x.shape[1]):
                k_grad[x_row, feat_idx] = 2 * gamma[0] * k * (x[x_row, feat_idx] - v[0, feat_idx])
