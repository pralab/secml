"""
.. module:: CKernelEuclidean
   :synopsis: Euclidean distance kernel.

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Angelo Sotgiu <angelo.sotgiu@unica.it>

"""
from sklearn import metrics

from secml.array import CArray
from secml.ml.kernels import CKernel


class CKernelEuclidean(CKernel):
    """Euclidean distance kernel.

    Given matrices X and RV, this is computed as the negative Euclidean dist.::

        K(x, rv) = -sqrt(dot(x, x) - 2 * dot(x, rv) + dot(rv, rv))

    for each pair of rows in X and in RV.
    If parameter squared is True (default False), sqrt() operation is avoided.

    Parameters
    ----------
    squared : bool, optional
        If True, return squared Euclidean distances. Default False.
    preprocess : CModule or None, optional
        Features preprocess to be applied to input data.
        Can be a CModule subclass. If None, input data is used as is.

    Attributes
    ----------
    class_type : 'euclidean'

    Examples
    --------
    >>> from secml.array import CArray
    >>> from secml.ml.kernels.c_kernel_euclidean import CKernelEuclidean

    >>> print(CKernelEuclidean().k(CArray([[1,2],[3,4]]), CArray([[10,20],[30,40]])))
    CArray([[-20.124612 -47.801674]
     [-17.464249 -45.      ]])

    >>> print(CKernelEuclidean().k(CArray([[1,2],[3,4]])))
    CArray([[0.       -2.828427]
     [-2.828427 0.      ]])

    """
    __class_type = 'euclidean'

    def __init__(self, squared=False, preprocess=None):
        self._squared = squared
        self._x_norm_squared = None
        self._rv_norm_squared = None
        super(CKernelEuclidean, self).__init__(preprocess=preprocess)

    @property
    def _grad_requires_forward(self):
        """Returns True as kernel is cached in the forward pass and then
        used by backward when computing the gradient."""
        return True

    @property
    def squared(self):
        """If True, squared Euclidean distances are computed."""
        return self._squared

    @squared.setter
    def squared(self, value):
        """Sets the squared parameter.

        Parameters
        ----------
        value : bool
            If True, squared Euclidean distances are computed.

        """
        self._squared = value

    @property
    def x_norm_squared(self):
        """Pre-computed dot-products of vectors in x
        (e.g., (x**2).sum(axis=1)).

        """
        return self._x_norm_squared

    @x_norm_squared.setter
    def x_norm_squared(self, value):
        """Sets the pre-computed dot-products of vectors in x.

        Parameters
        ----------
        value : CArray
            Pre-computed dot-products of vectors in x.

        """
        self._x_norm_squared = value

    @property
    def rv_norm_squared(self):
        """Pre-computed dot-products of vectors in rv
        (e.g., (rv**2).sum(axis=1)).

        """
        return self._rv_norm_squared

    @rv_norm_squared.setter
    def rv_norm_squared(self, value):
        """Sets the pre-computed dot-products of vectors in rv.

        Parameters
        ----------
        value : CArray
            Pre-computed dot-products of vectors in rv.

        """
        self._rv_norm_squared = value

    def _forward(self, x):
        """Compute this kernel as the negative Euclidean dist. between x and
        cached rv.

        Parameters
        ----------
        x : CArray
            Array of shape (n_x, n_features).

        Returns
        -------
        kernel : CArray
            Kernel between x and cached rv, shape (n_x, n_rv).

        """
        k = -CArray(metrics.pairwise.euclidean_distances(
            x.get_data(), self._rv.get_data(), squared=self._squared,
            X_norm_squared=self._x_norm_squared,
            Y_norm_squared=self._rv_norm_squared))
        self._cached_kernel = None if self._cached_x is None or self._squared \
            else k
        return k

    def _backward(self, w=None):
        """Compute the kernel gradient wrt cached vector 'x'.

        The gradient of Euclidean distance kernel is given by::

            dK(rv,x)/dx = - (rv - x) / k(rv,x)    if squared = False (default)
            dK(rv,x)/dx = 2 * (rv - x)        if squared = True

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

        if self._rv is None or (not self._squared
                                and self._cached_kernel is None):
            raise ValueError("Please run forward with caching=True first.")

        # Format of output array should be the same as cached x
        self._rv = self._rv.tosparse() if self._cached_x.issparse \
            else self._rv.todense()

        if self._squared is True:  # 2 * (rv - x)
            diff = (self._rv - self._cached_x)
            return 2 * diff if w is None else w.dot(2 * diff)

        diff = (self._rv - self._cached_x)

        k_grad = self._cached_kernel.T
        k_grad[k_grad == 0] = 1.0  # To avoid nans later

        # Casting the kernel to sparse if needed for efficient broadcasting
        if diff.issparse is True:
            k_grad = k_grad.tosparse()

        # - (rv - x) / k(rv,x)
        grad = -diff / k_grad
        grad = grad if w is None else w.dot(grad)

        # Casting to sparse if necessary
        return grad.tosparse() if diff.issparse else grad
