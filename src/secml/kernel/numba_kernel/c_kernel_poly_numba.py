"""
.. module:: KernelPolynomialNumba
   :synopsis: Polynomial kernel \w Numba optimization

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from numba import guvectorize

from prlib.array import CArray
from prlib.kernel.c_kernel_poly import CKernelPoly

from prlib.kernel.numba_kernel.numba_utils import sqrd_eucl_dense, dot_dense


class CKernelPolyNumba(CKernelPoly):
    """Polynomial Kernel with Numba optimization.

    Compute the Polynomial kernel between X and Y::

        K(x, y) = (coef0 + gamma * <x, y>)^degree

    for each pair of rows in X and in Y.

    Attributes
    ----------
    usenumba : True as current kernel has been optimized with Numba.
    cache_size : int, size of the cache used for kernel computation. Default 100.

    Parameters
    ----------
    degree : int
        Default is 2. Integer degree of the kernel.
    gamma : float
        Default is 1.0. This is a free parameter to be used for balancing.
    coef0 : float
        Default is 1.0. Free parameter used for trading off the influence
        of higher-order versus lower-order terms in the kernel.

    Examples
    --------
    >>> from prlib.array import CArray
    >>> from prlib.kernel.numba_kernel.c_kernel_poly_numba import CKernelPolyNumba

    >>> print CKernelPolyNumba(degree=3, gamma=0.001, coef0=2).k(CArray([[1,2],[3,4]]), CArray([[10,20],[30,40]]))
    CArray([[  8.615125   9.393931]
     [  9.393931  11.390625]])

    >>> print CKernelPolyNumba().k(CArray([[1,2],[3,4]]))
    CArray([[  36.  144.]
     [ 144.  676.]])

    """
    usenumba = True

    def _k(self, x, y):
        """Compute the Polynomial kernel between x and y.

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
        Polynomial kernel with Numba optimization is only available for
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
            return CArray(self._fast_poly(
                x_carray.tondarray(), y_carray.tondarray(),
                self.degree, self.gamma, self.coef0))

    def gradient(self, x, v):
        """Calculates Polynomial kernel gradient wrt vector 'v'.

        The gradient of Polynomial kernel is given by::

            dK(x[i],v)/dv =     x[i] * gamma * degree * k(x[i],v, degree-1)  if x[i] != v
                          = 2 * x[i] * gamma * degree * k(x[i],v, degree-1)  if x[i] == v

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
            Kernel gradient of u with respect to vector v. Array of
            shape (n_x, n_features) if n_x > 1, else a flattened
            array of shape (n_features, ).

        Notes
        -----
        Gradient of Polynomial kernel with Numba optimization is only available for
        the following data types:

        .. hlist::

         * `int32`, `int64`
         * `float32`, `float64`

        Examples
        --------
        >>> from prlib.array import CArray
        >>> from prlib.kernel.numba_kernel.c_kernel_poly_numba import CKernelPolyNumba

        >>> array = CArray([[15,25],[45,55]])
        >>> vector = CArray([2,5])
        >>> print CKernelPolyNumba(degree=3, gamma=1e-4, coef0=2).gradient(array, vector)
        CArray([[ 0.01828008  0.0304668 ]
         [ 0.05598899  0.06843098]])

        >>> print CKernelPolyNumba().gradient(vector, vector)
        CArray([ 240.  600.])

        """
        x_carray = CArray(x)
        v_carray = CArray(v)
        if x_carray.issparse is True or v_carray.issparse is True:
            return super(self.__class__, self).gradient(x, v)
        else:
            k_grad = self._gradient(x_carray.atleast_2d(),
                                    v_carray.atleast_2d())
            return k_grad if k_grad.shape[0] > 1 else k_grad.ravel()

    def _gradient(self, x, v):
        """Calculate Polynomial kernel gradient wrt vector 'v'.

        Parameters
        ----------
        x : CArray or array_like
            First array of shape (n_x, n_features).
        v : CArray or array_like
            Second array of shape (1, n_features).

        Returns
        -------
        kernel_gradient : CArray
            Kernel gradient of u with respect to vector v. Array of
            shape (1, n_features)

        See Also
        --------
        :meth:`.CKernelPolyNumba.gradient` : Gradient of Polynomial kernel with Numba optimization.

        """
        x_carray = CArray(x)
        v_carray = CArray(v)
        if x_carray.issparse is True or v_carray.issparse is True:
            return super(self.__class__, self)._gradient(x, v)
        else:
            return CArray(self._fast_poly_gradient(
                x_carray.tondarray(), v_carray.tondarray(),
                self.degree, self.gamma, self.coef0))

    @staticmethod
    @guvectorize(['void(int32[:,:], int32[:,:], int32[:], float32[:], float32[:], float32[:,:])',
                  'void(int64[:,:], int64[:,:], int64[:], float64[:], float64[:], float64[:,:])',
                  'void(float32[:,:], float32[:,:], int32[:], float32[:], float32[:], float32[:,:])',
                  'void(float64[:,:], float64[:,:], int64[:], float64[:], float64[:], float64[:,:])'],
                 '(m,n),(p,n),(),(),()->(m,p)', nopython=True)
    def _fast_poly(x, y, degree, gamma, coef0, k):

        for x_row in range(x.shape[0]):
            for y_row in range(y.shape[0]):

                # Computing (coef0 + gamma * x^T * y)^degree
                k[x_row, y_row] = (coef0[0] + gamma[0] * dot_dense(x, y, x_row, y_row)) ** degree[0]

    @staticmethod
    @guvectorize(['void(int32[:,:], int32[:,:], int32[:], float32[:], float32[:], float32[:,:])',
                  'void(int64[:,:], int64[:,:], int64[:], float64[:], float64[:], float64[:,:])',
                  'void(float32[:,:], float32[:,:], int32[:], float32[:], float32[:], float32[:,:])',
                  'void(float64[:,:], float64[:,:], int64[:], float64[:], float64[:], float64[:,:])'],
                 '(m,n),(p,n),(),(),()->(m,n)', nopython=True)
    def _fast_poly_gradient(x, v, degree, gamma, coef0, k_grad):

        for x_row in range(x.shape[0]):

            # Computing (coef0 + gamma * x^T * v)^degree
            k = (coef0[0] + gamma[0] * dot_dense(x, v, x_row, 0)) ** (degree[0] - 1)

            # Check if current x row and v are the same vector
            if sqrd_eucl_dense(x, v, x_row, 0) < 1e-8:
                # if x row and v are the same, return 2 * grad
                factor = 2.0
            else:
                factor = 1.0

            # Computing final kernel
            for feat_idx in range(x.shape[1]):
                k_grad[x_row, feat_idx] = factor * x[x_row, feat_idx] * k * gamma[0] * degree[0]
