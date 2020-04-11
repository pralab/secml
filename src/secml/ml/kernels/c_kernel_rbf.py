"""
.. py:module:: CKernelRBF
   :synopsis: Radial basis function (RBF) kernel

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from sklearn import metrics

from secml.array import CArray
from secml.ml.kernels import CKernel


class CKernelRBF(CKernel):
    """Radial basis function (RBF) kernel.

    Given matrices X and RV, this is computed by::

        K(x, rv) = exp(-gamma ||x-rv||^2)

    for each pair of rows in X and in RV.

    Parameters
    ----------
    gamma : float
        Default is 1.0. Equals to `-0.5 * sigma^-2` in the standard
        formulation of rbf kernel, it is a free parameter to be used
        for balancing.
    preprocess : CModule or None, optional
        Features preprocess to be applied to input data.
        Can be a CModule subclass. If None, input data is used as is.

    Attributes
    ----------
    class_type : 'rbf'

    Examples
    --------
    >>> from secml.array import CArray
    >>> from secml.ml.kernels.c_kernel_rbf import CKernelRBF

    >>> print(CKernelRBF(gamma=0.001).k(CArray([[1,2],[3,4]]), CArray([[10,20],[30,40]])))
    CArray([[0.666977 0.101774]
     [0.737123 0.131994]])

    >>> print(CKernelRBF().k(CArray([[1,2],[3,4]])))
    CArray([[1.000000e+00 3.354626e-04]
     [3.354626e-04 1.000000e+00]])

    """
    __class_type = 'rbf'

    def __init__(self, gamma=1.0, preprocess=None):

        # Using a float gamma to avoid dtype casting problems
        self.gamma = gamma
        super(CKernelRBF, self).__init__(preprocess=preprocess)

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
            Equals to `-0.5*sigma^-2` in the standard formulation of
            rbf kernel, is a free parameter to be used for balancing
            the computed metric.

        """
        self._gamma = float(gamma)

    def _forward(self, x):
        """Compute the rbf (gaussian) kernel between x and cached rv.

        Parameters
        ----------
        x : CArray or array_like
            Array of shape (n_x, n_features).

        Returns
        -------
        kernel : CArray
            Kernel between x and cached reference_samples, shape (n_x, n_rv).

        """
        k = CArray(metrics.pairwise.rbf_kernel(
            CArray(x).get_data(), CArray(self._rv).get_data(), self.gamma))
        self._cached_kernel = None if self._cached_x is None else k
        return k

    def _backward(self, w=None):
        """Calculate RBF kernel gradient wrt cached vector 'x'.

        The gradient of RBF kernel is given by::

            dK(rv,x)/dx = 2 * gamma * k(rv,x) * (rv - x)

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

        if self._rv is None or self._cached_kernel is None:
            raise ValueError("Please run forward with caching=True first.")

        # Format of output array should be the same as cached x
        self._rv = self._rv.tosparse() if self._cached_x.issparse \
            else self._rv.todense()

        k_grad = self._cached_kernel.T

        if w is not None:
            c = w.T * k_grad
            return CArray(2 * self.gamma * (c.T.dot(self._rv)
                                            - c.sum() * self._cached_x))
        else:
            diff = (self._rv - self._cached_x)
            # Casting the kernel to sparse if needed for efficient broadcasting
            if diff.issparse is True:
                k_grad = k_grad.tosparse()
            grad = CArray(2 * self.gamma * diff * k_grad)
            return grad if w is None else w.dot(grad)
