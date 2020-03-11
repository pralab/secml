"""
.. module:: CKernelHistIntersect
   :synopsis: Histogram Intersection kernel

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
import numpy as np

from secml.array import CArray
from secml.ml.kernels import CKernel


class CKernelHistIntersect(CKernel):
    """Histogram Intersection Kernel.

    Given matrices X and RV, this is computed by::

        K(x, rv) = sum_i ( min(x[i], rv[i]) )

    for each pair of rows in X and in RV.

    Attributes
    ----------
    class_type : 'hist-intersect'

    Examples
    --------
    >>> from secml.array import CArray
    >>> from secml.ml.kernels.c_kernel_histintersect import CKernelHistIntersect

    >>> print(CKernelHistIntersect().k(CArray([[1,2],[3,4]]), CArray([[10,20],[30,40]])))
    CArray([[3. 3.]
     [7. 7.]])

    >>> print(CKernelHistIntersect().k(CArray([[1,2],[3,4]])))
    CArray([[3. 3.]
     [3. 7.]])

    """
    __class_type = 'hist-intersect'

    def _forward(self, x):
        """Compute the histogram intersection kernel between x and cached rv.

        Parameters
        ----------
        x : CArray or array_like
            Array of shape (n_x, n_features).

        Returns
        -------
        kernel : CArray
            Kernel between x and cached rv. Array of shape (n_x, n_rv).

        """
        k = CArray.zeros(shape=(x.shape[0], self._rv.shape[0]))
        x_nd, rv_nd = x.tondarray(), self._rv.tondarray()

        if x.shape[0] <= self._rv.shape[0]:  # loop on the matrix with less samples
            # loop over samples in x, and compute x_i vs rv
            for i in range(k.shape[0]):
                k[i, :] = CArray(np.minimum(x_nd[i, :], rv_nd).sum(axis=1))
        else:
            # loop over samples in rv, and compute rv_j vs x
            for j in range(k.shape[1]):
                k[:, j] = CArray(np.minimum(x_nd, rv_nd[j, :]).sum(axis=1)).T

        return k

    def _backward(self, w=None):
        """Calculate Histogram Intersection kernel gradient wrt
        cached vector 'x'.

        The kernel is computed between each row of rv
        (denoted with rk) and x, as::
            sum_i ( min(rk[i], x[i]) )

        The gradient computed w.r.t. x is thus
        1 if x[i] < rk[i], and 0 elsewhere.

        Parameters
        ----------
        w : CArray of shape (1, n_rv) or None
            if CArray, it is pre-multiplied to the gradient
            of the module, as in standard reverse-mode autodiff.

        Returns
        -------
        kernel_gradient : CArray
            Kernel gradient of rv with respect to vector x,
            shape (n_rv, n_features) if n_rv > 1 and w is None,
            else (1, n_features).

        """
        # Checking if cached x is a vector
        if not self._cached_x.is_vector_like:
            raise ValueError(
                "kernel gradient can be computed only wrt vector-like arrays.")

        if self._rv is None:
            raise ValueError("Please run forward with caching=True or set"
                             "`rv` first.")

        if self._cached_x.issparse is True:
            # Broadcasting not supported for sparse arrays
            x_broadcast = self._cached_x.repmat(self._rv.shape[0], 1)
        else:  # Broadcasting is supported by design for dense arrays
            x_broadcast = self._cached_x

        grad = CArray.zeros(shape=self._rv.shape,
                            sparse=self._cached_x.issparse)
        grad[x_broadcast < self._rv] = 1  # TODO support from CArray still missing
        return grad if w is None else w.dot(grad)
