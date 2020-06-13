"""
.. module:: CKernelLinear
   :synopsis: Linear kernel

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml.array import CArray
from secml.ml.kernels import CKernel


class CKernelLinear(CKernel):
    """Linear kernel.

    Given matrices X and RV, this is computed by::

        K(x, rv) = x * rv^T

    for each pair of rows in X and in RV.

    Attributes
    ----------
    class_type : 'linear'

    Examples
    --------
    >>> from secml.array import CArray
    >>> from secml.ml.kernels.c_kernel_linear import CKernelLinear

    >>> print(CKernelLinear().k(CArray([[1,2],[3,4]]), CArray([[10,20],[30,40]])))
    CArray([[ 50. 110.]
     [110. 250.]])

    >>> print(CKernelLinear().k(CArray([[1,2],[3,4]])))
    CArray([[ 5. 11.]
     [11. 25.]])

    """
    __class_type = 'linear'

    def _forward(self, x):
        """Compute the linear kernel between x and cached rv.

        Parameters
        ----------
        x : CArray or array_like
            Array of shape (n_x, n_features).

        Returns
        -------
        kernel : CArray
            Kernel between x and cached rv. Array of shape (n_x, n_rv).

        """
        return CArray(x.dot(self.rv.T))

    def _backward(self, w=None):
        """Calculate Linear kernel gradient wrt cached vector 'x'.

        The gradient of Linear kernel is given by::

            dK(rv,x)/dx = rv

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

        # Format of output array should be the same as rv
        grad = self._rv.deepcopy()
        grad = grad.tosparse() if self._cached_x.issparse else grad.todense()
        return grad if w is None else w.dot(grad)
