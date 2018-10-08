"""
.. module:: KernelEuclidean
   :synopsis: Euclidean distances kernel.

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from sklearn import metrics

from secml.array import CArray
from secml.kernel import CKernel


class CKernelEuclidean(CKernel):
    """Euclidean distances kernel.

    Given matrices X and Y, this is computed by::

        K(x, y) = ||x-y||^2

    for each pair of rows in X and in Y.

    Examples
    --------
    >>> from secml.array import CArray
    >>> from secml.kernel.c_kernel_euclidean import CKernelEuclidean

    >>> print CKernelEuclidean().k(CArray([[1,2],[3,4]]), CArray([[10,20],[30,40]]))
    CArray([[ 20.1246118   47.80167361]
     [ 17.4642492   45.        ]])

    >>> print CKernelEuclidean().k(CArray([[1,2],[3,4]]))
    CArray([[ 0.          2.82842712]
     [ 2.82842712  0.        ]])

    """
    class_type = 'euclidean'

    def __init__(self, cache_size=100):

        super(CKernelEuclidean, self).__init__(cache_size=cache_size)

    def _k(self, x, y):
        """Compute the euclidean kernel between x and y.

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
        :meth:`.CKernel.k` : Main computation interface for kernels.

        """
        return CArray(metrics.pairwise.pairwise_distances(
            CArray(x).get_data(), CArray(y).get_data(), metric='euclidean'))

    def _gradient(self, u, v):
        """Calculate Euclidean kernel gradient wrt vector 'v'."""
        raise NotImplementedError(
            "Gradient of Euclidean kernel is not available.")
