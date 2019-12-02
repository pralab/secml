"""
.. module:: CKernelChebyshevDistance
   :synopsis: Chebyshev distance kernel

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>


"""
from sklearn import metrics

from secml.array import CArray
from secml.ml.kernel import CKernel


class CKernelChebyshevDistance(CKernel):
    """Chebyshev distance kernel.

    Given matrices X and Y, this is computed as::

        K(x, y) = max(|x - y|)

    for each pair of rows in X and in Y.

    Attributes
    ----------
    class_type : 'chebyshev-dist'

    Examples
    --------
    >>> from secml.array import CArray
    >>> from secml.ml.kernel.c_kernel_chebyshev_distance import CKernelChebyshevDistance

    >>> print(CKernelChebyshevDistance().k(CArray([[1,2],[3,4]]), CArray([[5,6],[7,8]])))
    CArray([[-4. -6.]
     [-2. -4.]])

    >>> print(CKernelChebyshevDistance().k(CArray([[1,2],[3,4]])))
    CArray([[0. -2.]
     [-2. 0.]])

    """
    __class_type = 'chebyshev-dist'

    def _k(self, x, y):
        """Compute (negative) Chebyshev distances between x and y.

        Parameters
        ----------
        x : CArray or array_like
            First array of shape (n_x, n_features).
        y : CArray or array_like
            Second array of shape (n_y, n_features).

        Returns
        -------
        kernel : CArray
            Kernel between x and y, shape (n_x, n_y).

        See Also
        --------
        :meth:`CKernel.k` : Main computation interface for kernels.

        """
        if x.issparse is True or y.issparse is True:
            raise TypeError(
                "Chebyshev Kernel not available for sparse data."
                "See `sklearn.metrics.pairwise_distances`.")

        return -CArray(metrics.pairwise.pairwise_distances(
            x.get_data(), y.get_data(), metric='chebyshev'))

    def _gradient(self, u, v):
        """Calculate gradients of Chebyshev kernel wrt vector 'v'.

        The gradient of the negative Chebyshev distance is given by::

            dK(u,v)/dv =  -sign(u-v)

        Parameters
        ----------
        u : CArray
            First array of shape (nx, n_features).
        v : CArray
            Second array of shape (1, n_features).

        Returns
        -------
        kernel_gradient : CArray
            Kernel gradient of u with respect to vector v,
            shape (nx, n_features).

        See Also
        --------
        :meth:`CKernel.gradient` : Gradient computation interface for kernels.

        """
        diff = u - v
        m = abs(diff).max(axis=1)  # extract m from each row
        grad = CArray.zeros(shape=diff.shape, sparse=v.issparse)
        grad[diff >= m] = 1  # this correctly broadcasts per-row comparisons
        grad[diff <= -m] = -1
        return grad
