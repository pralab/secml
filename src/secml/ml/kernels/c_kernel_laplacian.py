"""
.. module:: CKernelLaplacian
   :synopsis: Laplacian kernel

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>

"""
from sklearn import metrics

from secml.array import CArray
from secml.ml.kernels import CKernel


class CKernelLaplacian(CKernel):
    """Laplacian Kernel.

    Given matrices X and RV, this is computed by::

        K(x, rv) = exp(-gamma |x-rv|)

    for each pair of rows in X and in RV.

    Parameters
    ----------
    gamma : float
        Default is 1.0.
    preprocess : CModule or None, optional
        Features preprocess to be applied to input data.
        Can be a CModule subclass. If None, input data is used as is.

    Attributes
    ----------
    class_type : 'laplacian'

    Examples
    --------
    >>> from secml.array import CArray
    >>> from secml.ml.kernels.c_kernel_laplacian import CKernelLaplacian

    >>> print(CKernelLaplacian(gamma=0.01).k(CArray([[1,2],[3,4]]), CArray([[10,0],[0,40]])))
    CArray([[0.895834 0.677057]
     [0.895834 0.677057]])

    >>> print(CKernelLaplacian().k(CArray([[1,2],[3,4]])))
    CArray([[1.       0.018316]
     [0.018316 1.      ]])

    """
    __class_type = 'laplacian'

    def __init__(self, gamma=1.0, preprocess=None):

        # Using a float gamma to avoid dtype casting problems
        self.gamma = gamma
        super(CKernelLaplacian, self).__init__(preprocess=preprocess)

    @property
    def _grad_requires_forward(self):
        """Returns True as kernel is cached in the forward pass and then
        used by backward when computing the gradient."""
        return True

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

    def _forward(self, x):
        """Compute the Laplacian kernel between x and cached rv.

        Parameters
        ----------
        x : CArray
            Array of shape (n_x, n_features).

        Returns
        -------
        kernel : CArray
            Kernel between x and cached rv, shape (n_x, n_rv).

        """
        k = CArray(metrics.pairwise.laplacian_kernel(
            CArray(x).get_data(), CArray(self._rv).get_data(),
            gamma=self.gamma))
        self._cached_kernel = None if self._cached_x is None else k
        return k

    def _backward(self, w):
        """Calculate Laplacian kernel gradient wrt vector 'x'.

        The gradient of Laplacian kernel is given by::

            dK(rv,x)/dx = gamma * k(rv,x) * sign(rv - x)

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
        if not self._cached_x.is_vector_like or self._cached_x.shape[0] > 1:
            raise ValueError(
                "kernel gradient can be computed only wrt arrays with shape "
                "(1, n_features).")

        if self._rv is None or self._cached_kernel is None:
            raise ValueError("Please run forward with caching=True first.")

        # Format of output array should be the same as x
        rv = self._rv.tosparse() if self._cached_x.issparse \
            else self._rv.todense()

        diff = (rv - self._cached_x)

        k_grad = self._cached_kernel.T

        # Casting the kernel to sparse if needed for efficient broadcasting
        if diff.issparse is True:
            k_grad = k_grad.tosparse()

        grad = self.gamma * k_grad * diff.sign()
        return grad if w is None else w.dot(grad)
