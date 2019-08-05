"""
.. module:: CKernelHistIntersect
   :synopsis: Histogram Intersection kernel

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from sklearn import metrics
import numpy as np

from secml.array import CArray
from secml.ml.kernel import CKernel


class CKernelHistIntersect(CKernel):
    """Histogram Intersection Kernel.

    Given matrices X and Y, this is computed by::

        K(x, y) = sum^n_i ( min(x[i], y[i]) )

    for each pair of rows in X and in Y.

    Attributes
    ----------
    class_type : 'hist-intersect'

    Parameters
    ----------
    batch_size : int or None, optional
        Size of the batch used for kernel computation. Default None.

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

    # TODO: ADD SPARSE SUPPORT
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
        x_carray = CArray(x)
        y_carray = CArray(y)
        if x_carray.issparse is True or y_carray.issparse is True:
            raise TypeError(
                "Histogram Intersection Kernel not available for sparse data.")

        # Defining kernel value for each pair
        def hint(s1, s2):
            # Working with ndarrays as will be called by pairwise_kernels
            return np.minimum(s1, s2).sum()

        # Calling sklearn pairwise_kernel to compute pairwise fast and clean :)
        return CArray(metrics.pairwise.pairwise_kernels(
            x_carray.get_data(), y_carray.get_data(), metric=hint))

    def _gradient(self, u, v):
        """Calculate Histogram Intersection kernel gradient wrt vector 'v'.

        .. warning::

           Gradient for Histogram Intersection kernel is not analytically
           computable so is currently not available.

        See Also
        --------
        :meth:`.CKernel.gradient` : Gradient computation interface for kernels.

        """
        raise NotImplementedError(
            "Gradient of Histogram Intersection Kernel is not available.")
