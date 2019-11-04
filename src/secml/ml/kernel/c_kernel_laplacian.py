"""
.. module:: CKernelLaplacian
   :synopsis: Laplacian kernel

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>

"""
from sklearn import metrics

from secml.array import CArray
from secml.ml.kernel import CKernel


class CKernelLaplacian(CKernel):
    """Laplacian Kernel.

    Given matrices X and Y, this is computed by::

        K(x, y) = exp(-gamma |x-y|)

    for each pair of rows in X and in Y.

    Attributes
    ----------
    class_type : 'laplacian'

    Parameters
    ----------
    gamma : float
        Default is 1.0.

    Examples
    --------
    >>> from secml.array import CArray
    >>> from secml.ml.kernel.c_kernel_laplacian import CKernelLaplacian

    >>> print(CKernelLaplacian(gamma=0.01).k(CArray([[1,2],[3,4]]), CArray([[10,0],[0,40]])))
    CArray([[0.895834 0.677057]
     [0.895834 0.677057]])

    >>> print(CKernelLaplacian().k(CArray([[1,2],[3,4]])))
    CArray([[1.       0.018316]
     [0.018316 1.      ]])

    """
    __class_type = 'laplacian'

    def __init__(self, gamma=1.0):

        # Using a float gamma to avoid dtype casting problems
        self.gamma = gamma

    @property
    def gamma(self):
        """Gamma parameter."""
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        """Sets gamma parameter.

        Parameters
        ----------
        gamma : float
            Equals to `sigma^-1` in the standard formulation of
            Laplacian kernel, is a free parameter to be used
            to balance the computed metric.

        """
        self._gamma = float(gamma)

    def _k(self, x, y):
        """Compute the Laplacian kernel between x and y.

        The gradient of Laplacian kernel is given by::

            dK(x,v)/dv = gamma * k(x,v) * sign(x - v)

        Parameters
        ----------
        x : CArray
            First array of shape (n_x, n_features).
        y : CArray
            Second array of shape (n_y, n_features).

        Returns
        -------
        kernel : CArray
            Kernel between x and y, shape (n_x, n_y).

        See Also
        --------
        :meth:`CKernel.k` : Main computation interface for kernels.

        """
        return CArray(metrics.pairwise.laplacian_kernel(
            CArray(x).get_data(), CArray(y).get_data(), gamma=self.gamma))

    def _gradient(self, x, v):
        """Calculate Laplacian kernel gradient wrt vector 'v'.

        The gradient of Laplacian kernel is given by::

            dK(x,v)/dv =  gamma * k(x,v) * sign(x - v)

        Parameters
        ----------
        x : CArray
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
        if v.shape[0] > 1:
            raise ValueError(
                "2nd array must have shape shape (1, n_features).")

        if v.issparse is True:
            # Broadcasting not supported for sparse arrays
            v_broadcast = v.repmat(x.shape[0], 1)
        else:  # Broadcasting is supported by design for dense arrays
            v_broadcast = v

        # Format of output array should be the same as v
        x = x.tosparse() if v.issparse else x.todense()

        diff = (x - v_broadcast)

        k_grad = self._k(x, v)
        # Casting the kernel to sparse if needed for efficient broadcasting
        if diff.issparse is True:
            k_grad = k_grad.tosparse()

        return self.gamma * k_grad * diff.sign()
