"""
.. module:: CKernelChebyshevDistance
   :synopsis: Chebyshev distance kernel

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Angelo Sotgiu <angelo.sotgiu@unica.it>

"""
from sklearn import metrics

from secml.array import CArray
from secml.ml.kernels import CKernel


class CKernelChebyshevDistance(CKernel):
    """Chebyshev distance kernel.

    Given matrices X and RV, this is computed as::

        K(x, rv) = max(|x - rv|)

    for each pair of rows in X and in RV.

    Attributes
    ----------
    class_type : 'chebyshev-dist'

    Examples
    --------
    >>> from secml.array import CArray
    >>> from secml.ml.kernels import CKernelChebyshevDistance

    >>> x = CArray([[1,2],[3,4]])
    >>> v = CArray([[5,6],[7,8]])
    >>> print(CKernelChebyshevDistance().k(x,v))
    CArray([[-4. -6.]
     [-2. -4.]])

    >>> print(CKernelChebyshevDistance().k(x))
    CArray([[-0. -2.]
     [-2. -0.]])

    """
    __class_type = 'chebyshev-dist'

    def _forward(self, x):
        """Compute (negative) Chebyshev distances between x and cached rv.

        Parameters
        ----------
        x : CArray or array_like
            Array of shape (n_x, n_features).

        Returns
        -------
        kernel : CArray
            Kernel between x and cached rv, shape (n_x, n_rv).

        """
        if x.issparse is True or self._rv.issparse is True:
            raise TypeError(
                "Chebyshev Kernel not available for sparse data."
                "See `sklearn.metrics.pairwise_distances`.")

        return -CArray(metrics.pairwise.pairwise_distances(
            x.get_data(), self._rv.get_data(),
            metric='chebyshev'))

    def _backward(self, w=None):
        """Calculate gradients of Chebyshev kernel wrt cached vector 'x'.

        The gradient of the negative Chebyshev distance is given by::

            dK(rv,x)/dx = - sign(rv - x)

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
        # checking if cached x is a vector
        if not self._cached_x.is_vector_like:
            raise ValueError(
                "kernel gradient can be computed only wrt vector-like arrays.")

        if self._rv is None:
            raise ValueError(
                "Please run forward with caching=True or set `rv` first.")

        if self._cached_x.issparse is True or self._rv.issparse is True:
            raise TypeError(
                "Chebyshev Kernel not available for sparse data."
                "See `sklearn.metrics.pairwise_distances`.")

        diff = self._rv - self._cached_x
        m = abs(diff).max(axis=1)  # extract m from each row
        grad = CArray.zeros(shape=diff.shape)
        grad[diff >= m] = 1  # this correctly broadcasts per-row comparisons
        grad[diff <= -m] = -1
        return grad if w is None else w.dot(grad)
