"""
.. module:: KernelEuclidean
   :synopsis: Euclidean distances kernel.

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from sklearn import metrics

from secml.array import CArray
from secml.ml.kernel import CKernel


class CKernelEuclidean(CKernel):
    """Euclidean distances kernel.

    Given matrices X and Y, this is computed by::

        K(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))

    for each pair of rows in X and in Y.
    If parameter squared is True (default False), sqrt() operation is avoided.

    Examples
    --------
    >>> from secml.array import CArray
    >>> from secml.ml.kernel.c_kernel_euclidean import CKernelEuclidean

    >>> print CKernelEuclidean().k(CArray([[1,2],[3,4]]), CArray([[10,20],[30,40]]))
    CArray([[ 20.124612  47.801674]
     [ 17.464249  45.      ]])

    >>> print CKernelEuclidean().k(CArray([[1,2],[3,4]]))
    CArray([[ 0.        2.828427]
     [ 2.828427  0.      ]])

    """
    class_type = 'euclidean'

    def __init__(self, cache_size=100):

        super(CKernelEuclidean, self).__init__(cache_size=cache_size)

    def _k(self, x, y, squared=False,
           x_norm_squared=None, y_norm_squared=None):
        """Compute the euclidean kernel between x and y.

        Parameters
        ----------
        x : CArray
            First array of shape (n_x, n_features).
        y : CArray
            Second array of shape (n_y, n_features).
        squared : bool, optional
            If True, return squared Euclidean distances. Default False
        x_norm_squared : CArray or None, optional
            Pre-computed dot-products of vectors in x (e.g., (x**2).sum(axis=1)).
        y_norm_squared : CArray or None, optional
            Pre-computed dot-products of vectors in y (e.g., (y**2).sum(axis=1)).

        Returns
        -------
        kernel : CArray
            Kernel between x and y, shape (n_x, n_y).

        See Also
        --------
        :meth:`.CKernel.k` : Main computation interface for kernels.

        """
        return CArray(metrics.pairwise.euclidean_distances(
            x.get_data(), y.get_data(), squared=squared,
            X_norm_squared=x_norm_squared, Y_norm_squared=y_norm_squared))

    def _gradient(self, u, v):
        """Calculate Euclidean kernel gradient wrt vector 'v'."""
        raise NotImplementedError(
            "Gradient of Euclidean kernel is not available.")
