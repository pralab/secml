"""
.. module:: CKernelLinear
   :synopsis: Linear kernel

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from sklearn import metrics

from secml.array import CArray
from secml.ml.kernel import CKernel


class CKernelLinear(CKernel):
    """Linear kernel.

    Given matrices X and Y, this is computed by::

        K(x, y) = x * y^T

    for each pair of rows in X and in Y.

    Attributes
    ----------
    class_type : 'linear'

    Parameters
    ----------
    batch_size : int or None, optional
        Size of the batch used for kernel computation. Default None.

        .. deprecated:: 0.10

    Examples
    --------
    >>> from secml.array import CArray
    >>> from secml.ml.kernel.c_kernel_linear import CKernelLinear

    >>> print(CKernelLinear().k(CArray([[1,2],[3,4]]), CArray([[10,20],[30,40]])))
    CArray([[ 50. 110.]
     [110. 250.]])

    >>> print(CKernelLinear().k(CArray([[1,2],[3,4]])))
    CArray([[ 5. 11.]
     [11. 25.]])

    """
    __class_type = 'linear'

    def _k(self, x, y):
        """Compute the linear kernel between x and y.

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
        :meth:`CKernel.k` : Main computation interface for kernels.

        """
        return CArray(metrics.pairwise.linear_kernel(
            CArray(x).get_data(), CArray(y).get_data()))

    def _gradient(self, u, v):
        """Calculate Linear kernel gradient wrt vector 'v'.

        The gradient of Linear kernel is given by::

            dK(u,v)/dv = u

        Parameters
        ----------
        u : CArray or array_like
            First array of shape (nx, n_features).
        v : CArray or array_like
            Second array of shape (1, n_features).

        Returns
        -------
        kernel_gradient : CArray
            Kernel gradient of u with respect to vector v,
            shape (1, n_features).

        See Also
        --------
        :meth:`CKernel.gradient` : Gradient computation interface for kernels.

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.ml.kernel.c_kernel_linear import CKernelLinear

        >>> array = CArray([[15, 25], [45, 55]])
        >>> vector = CArray([2, 5])
        >>> print(CKernelLinear().gradient(array, vector))
        CArray([[15 25]
         [45 55]])

        >>> print(CKernelLinear().gradient(vector, vector))
        CArray([ 4 10])

        """
        # Format of output array should be the same as v
        grad = u.deepcopy()
        grad = grad.tosparse() if v.issparse else grad.todense()
        return grad
