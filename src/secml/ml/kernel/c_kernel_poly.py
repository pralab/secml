"""
.. module:: CKernelPoly
   :synopsis: Polynomial kernel

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from sklearn import metrics

from secml.array import CArray
from secml.ml.kernel import CKernel


class CKernelPoly(CKernel):
    """Polynomial kernel.

    Given matrices X and Y, this is computed by::

        K(x, y) = (coef0 + gamma * <x, y>)^degree

    for each pair of rows in X and in Y.

    Attributes
    ----------
    class_type : 'poly'

    Parameters
    ----------
    degree : int, optional
        Kernel degree. Default 2.
    gamma : float, optional
        Free parameter to be used for balancing. Default 1.0.
    coef0 : float, optional
        Free parameter used for trading off the influence of higher-order
        versus lower-order terms in the kernel. Default 1.0.
    batch_size : int or None, optional
        Size of the batch used for kernel computation. Default None.

    Examples
    --------
    >>> from secml.array import CArray
    >>> from secml.ml.kernel.c_kernel_poly import CKernelPoly

    >>> print(CKernelPoly(degree=3, gamma=0.001, coef0=2).k(CArray([[1,2],[3,4]]), CArray([[10,20],[30,40]])))
    CArray([[ 8.615125  9.393931]
     [ 9.393931 11.390625]])

    >>> print(CKernelPoly().k(CArray([[1,2],[3,4]])))
    CArray([[ 36. 144.]
     [144. 676.]])

    """
    __class_type = 'poly'
    
    def __init__(self, degree=2, gamma=1.0, coef0=1.0, batch_size=None):

        super(CKernelPoly, self).__init__(batch_size=batch_size)

        # kernel parameters
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        
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

    def _k(self, x, y):
        """Compute the polynomial kernel between x and y.

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
        return CArray(metrics.pairwise.polynomial_kernel(
            CArray(x).get_data(), CArray(y).get_data(),
            self.degree, self.gamma, self.coef0))

    # TODO: check for high gamma, we may have uncontrolled behavior (too high values)
    def _gradient(self, u, v):
        """Calculate Polynomial kernel gradient wrt vector 'v'.

        The gradient of Polynomial kernel is given by::

            dK(u,v)/dv =     u * gamma * degree * k(u,v, degree-1)  if u != v
                       = 2 * u * gamma * degree * k(u,v, degree-1)  if u == v

        Parameters
        ----------
        u : CArray or array_like
            First array of shape (1, n_features).
        v : CArray or array_like
            Second array of shape (1, n_features).

        Returns
        -------
        kernel_gradient : CArray
            Kernel gradient of u with respect to vector v,
            shape (1, n_features).

        See Also
        --------
        :meth:`.CKernel.gradient` : Common gradient computation interface for kernels.

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.ml.kernel.c_kernel_poly import CKernelPoly

        >>> array = CArray([[15,25],[45,55]])
        >>> vector = CArray([2,5])
        >>> print(CKernelPoly(degree=3, gamma=1e-4, coef0=2).gradient(array, vector))
        CArray([[0.01828  0.030467]
         [0.055989 0.068431]])

        >>> print(CKernelPoly().gradient(vector, vector))
        CArray([240. 600.])

        """
        u_carray = CArray(u)
        v_carray = CArray(v)
        if u_carray.shape[0] + v_carray.shape[0] > 2:
            raise ValueError(
                "Both input arrays must be 2-Dim of shape (1, n_features).")

        k = CArray(metrics.pairwise.polynomial_kernel(
            u_carray.get_data(), v_carray.get_data(),
            self.degree-1, self.gamma, self.coef0))

        # Format of output array should be the same as v
        if v_carray.issparse:
            u_carray = u_carray.tosparse()
            # Casting the kernel to sparse for efficient broadcasting
            k = k.tosparse()
        else:
            u_carray = u_carray.todense()

        k_grad = u_carray * k.ravel() * self.gamma * self.degree
        return k_grad * 2 if (u_carray - v_carray).norm() < 1e-8 else k_grad
