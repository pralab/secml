"""
.. module:: CKernelHistIntersect
   :synopsis: Histogram Intersection kernel

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from sklearn import metrics
import numpy as np

from secml.array import CArray
from secml.ml.kernel import CKernel


class CKernelHistIntersect(CKernel):
    """Histogram Intersection Kernel.

    Given matrices X and Y, this is computed by::

        K(x, y) = sum_i ( min(x[i], y[i]) )

    for each pair of rows in X and in Y.

    Attributes
    ----------
    class_type : 'hist-intersect'

    Examples
    --------
    >>> from secml.array import CArray
    >>> from secml.ml.kernel.c_kernel_histintersect import CKernelHistIntersect

    >>> print(CKernelHistIntersect().k(CArray([[1,2],[3,4]]), CArray([[10,20],[30,40]])))
    CArray([[3. 3.]
     [7. 7.]])

    >>> print(CKernelHistIntersect().k(CArray([[1,2],[3,4]])))
    CArray([[3. 3.]
     [3. 7.]])

    """
    __class_type = 'hist-intersect'

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

        See Also
        --------
        :meth:`.CKernel.k` : Common computation interface for kernels.

        """
        x = CArray(x).atleast_2d()
        y = x if y is None else CArray(y).atleast_2d()

        k = CArray.zeros(shape=(x.shape[0], y.shape[0]))

        x_nd, y_nd = x.tondarray(), y.tondarray()

        if x.shape[0] <= y.shape[0]:  # loop on the matrix with less samples
            # loop over samples in x, and compute x_i vs y
            for i in range(k.shape[0]):
                k[i, :] = CArray(np.minimum(x_nd[i, :], y_nd).sum(axis=1))
        else:
            # loop over samples in y, and compute y_j vs x
            for j in range(k.shape[1]):
                k[:, j] = CArray(np.minimum(x_nd, y_nd[j, :]).sum(axis=1)).T

        return k

    def _gradient(self, u, v):
        """Calculate Histogram Intersection kernel gradient wrt vector 'v'.

        The kernel is computed between each row of u
        (denoted with uk) and v, as::
            sum_i ( min(uk[i], v[i]) )

        The gradient computed w.r.t. v is thus
        1 if v[i] < uk[i], and 0 elsewhere.

        Parameters
        ----------
        u : CArray or array_like
            First array of shape (n_x, n_features).
        v : CArray or array_like
            Second array of shape (1, n_features).

        Returns
        -------
        gradient : CArray
            dK(u,v)/dv. Array of shape (n_x, n_features).

        See Also
        --------
        :meth:`.CKernel.gradient` : Gradient computation interface for kernels.

        """
        if v.issparse is True:
            # Broadcasting not supported for sparse arrays
            v_broadcast = v.repmat(u.shape[0], 1)
        else:  # Broadcasting is supported by design for dense arrays
            v_broadcast = v

        grad = CArray.zeros(shape=u.shape, sparse=v.issparse)
        grad[v_broadcast < u] = 1  # TODO support from CArray still missing
        return grad
