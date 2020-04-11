"""
.. module:: CKernelPoly
   :synopsis: Polynomial kernel

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from sklearn import metrics

from secml.array import CArray
from secml.ml.kernels import CKernel


class CKernelPoly(CKernel):
    """Polynomial kernel.

    Given matrices X and RV, this is computed by::

        K(x, rv) = (coef0 + gamma * <x, rv>)^degree

    for each pair of rows in X and in RV.

    Parameters
    ----------
    degree : int, optional
        Kernel degree. Default 2.
    gamma : float, optional
        Free parameter to be used for balancing. Default 1.0.
    coef0 : float, optional
        Free parameter used for trading off the influence of higher-order
        versus lower-order terms in the kernel. Default 1.0.
    preprocess : CModule or None, optional
        Features preprocess to be applied to input data.
        Can be a CModule subclass. If None, input data is used as is.

    Attributes
    ----------
    class_type : 'poly'

    Examples
    --------
    >>> from secml.array import CArray
    >>> from secml.ml.kernels.c_kernel_poly import CKernelPoly

    >>> print(CKernelPoly(degree=3, gamma=0.001, coef0=2).k(CArray([[1,2],[3,4]]), CArray([[10,20],[30,40]])))
    CArray([[ 8.615125  9.393931]
     [ 9.393931 11.390625]])

    >>> print(CKernelPoly().k(CArray([[1,2],[3,4]])))
    CArray([[ 36. 144.]
     [144. 676.]])

    """
    __class_type = 'poly'

    def __init__(self, degree=2, gamma=1.0, coef0=1.0, preprocess=None):

        # kernel parameters
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        super(CKernelPoly, self).__init__(preprocess=preprocess)

    @property
    def degree(self):
        """Degree parameter."""
        return self._degree

    @degree.setter
    def degree(self, degree):
        """Sets degree parameter.

        Parameters
        ----------
        degree : int
            Default is 2. Integer degree of the kernel.

        """
        self._degree = int(degree)

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
            Default is 1.0. This is a free parameter to be used for balancing.

        """
        self._gamma = float(gamma)

    @property
    def coef0(self):
        """Coef0 parameter."""
        return self._coef0

    @coef0.setter
    def coef0(self, coef0):
        """Sets coef0 parameter.

        Parameters
        ----------
        coef0 : float
            Default is 1.0. Free parameter used for trading off the influence
            of higher-order versus lower-order terms in the kernel.

        """
        self._coef0 = float(coef0)

    def _forward(self, x):
        """Compute the polynomial kernel between x and cached rv.

        Parameters
        ----------
        x : CArray or array_like
            Array of shape (n_x, n_features).

        Returns
        -------
        kernel : CArray
            Kernel between x and rv. Array of shape (n_x, n_rv).

        """
        return CArray(metrics.pairwise.polynomial_kernel(
            CArray(x).get_data(), CArray(self._rv).get_data(),
            self.degree, self.gamma, self.coef0))

    # TODO: check for high gamma,
    #  we may have uncontrolled behavior (too high values)
    def _backward(self, w=None):
        """Calculate Polynomial kernel gradient wrt cached vector 'x'.

        The gradient of Polynomial kernel is given by::

            dK(rv,x)/dy = rv * gamma * degree * k(rv,x, degree-1)

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

        k = CArray(metrics.pairwise.polynomial_kernel(
            self._rv.get_data(), self._cached_x.get_data(),
            self.degree - 1, self.gamma, self.coef0))

        # Format of output array should be the same as cached x
        if self._cached_x.issparse:
            rv = self._rv.tosparse()
            # Casting the kernel to sparse for efficient broadcasting
            k = k.tosparse()
        else:
            rv = self._rv.todense()

        grad = rv * k * self.gamma * self.degree
        return grad if w is None else w.dot(grad)
