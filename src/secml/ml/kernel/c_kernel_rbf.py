"""
.. py:module:: CKernelRBF
   :synopsis: Radial basis function (RBF) kernel

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from sklearn import metrics

from secml.array import CArray
from secml.ml.kernel import CKernel


class CKernelRBF(CKernel):
    """Radial basis function (RBF) kernel.

    Given matrices X and Y, this is computed by::

        K(x, y) = exp(-gamma ||x-y||^2)

    for each pair of rows in X and in Y.

    Attributes
    ----------
    class_type : 'rbf'

    Parameters
    ----------
    gamma : float
        Default is 1.0. Equals to `-0.5 * sigma^-2` in the standard
        formulation of rbf kernel, it is a free parameter to be used
        for balancing.
    batch_size : int or None, optional
        Size of the batch used for kernel computation. Default None.

    Examples
    --------
    >>> from secml.array import CArray
    >>> from secml.ml.kernel.c_kernel_rbf import CKernelRBF

    >>> print(CKernelRBF(gamma=0.001).k(CArray([[1,2],[3,4]]), CArray([[10,20],[30,40]])))
    CArray([[0.666977 0.101774]
     [0.737123 0.131994]])

    >>> print(CKernelRBF().k(CArray([[1,2],[3,4]])))
    CArray([[1.000000e+00 3.354626e-04]
     [3.354626e-04 1.000000e+00]])

    """
    __class_type = 'rbf'

    def __init__(self, gamma=1.0, batch_size=None):

        super(CKernelRBF, self).__init__(batch_size=batch_size)

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
            Equals to `-0.5*sigma^-2` in the standard formulation of
            rbf kernel, is a free parameter to be used for balancing
            the computed metric.

        """
        self._gamma = float(gamma)

    def _k(self, x, y):
        """Compute the rbf (gaussian) kernel between x and y.

        Parameters
        ----------
        x : CArray or array_like
            First array of shape (n_x, n_features).
        y : CArray or array_like
            Second array of shape (n_y, n_features).

        Returns
        -------
        kernel : CArray
            Kernel between x and y, shape (n_x, n_y).

        See Also
        --------
        :meth:`CKernel.k` : Main computation interface for kernels.

        """
        return CArray(metrics.pairwise.rbf_kernel(
            CArray(x).get_data(), CArray(y).get_data(), self.gamma))

    def gradient(self, x, v):
        """Calculates RBF kernel gradient wrt vector 'v'.

        The gradient of RBF kernel is given by::

            dK(x,v)/dv = 2 * gamma * k(x,v) * (x - v)

        Parameters
        ----------
        x : CArray or array_like
            First array of shape (n_x, n_features).
        v : CArray or array_like
            Second array of shape (n_features, ) or (1, n_features).

        Returns
        -------
        kernel_gradient : CArray
            Kernel gradient of x with respect to vector v. Array of
            shape (n_x, n_features) if n_x > 1, else a flattened
            array of shape (n_features, ).

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.ml.kernel.c_kernel_rbf import CKernelRBF

        >>> array = CArray([[15,25],[45,55]])
        >>> vector = CArray([2,5])
        >>> print(CKernelRBF(gamma=1e-4).gradient(array, vector))
        CArray([[0.002456 0.003779]
         [0.005567 0.006473]])

        >>> print(CKernelRBF().gradient(vector, vector))
        CArray([0. 0.])

        """
        x_carray = CArray(x).atleast_2d()
        v_carray = CArray(v).atleast_2d()

        # Checking if second array is a vector
        if v_carray.shape[0] > 1:
            raise ValueError(
                "kernel gradient can be computed only wrt vector-like arrays.")

        grad = self._gradient(x_carray, v_carray)
        return grad.ravel() if x_carray.shape[0] == 1 else grad

    def _gradient(self, u, v):
        """Calculate RBF kernel gradient wrt vector 'v'.

        The gradient of RBF kernel is given by::

            dK(u,v)/dv = 2 * gamma * k(u,v) * (u - v)

        Parameters
        ----------
        u : CArray or array_like
            First array of shape (n_x, n_features).
        v : CArray or array_like
            Second array of shape (n_features, ) or (1, n_features).

        Returns
        -------
        kernel_gradient : CArray
            Kernel gradient of u with respect to vector v,
            shape (1, n_features).

        See Also
        --------
        :meth:`CKernel.gradient` : Gradient computation interface for kernels.

        """
        u_carray = CArray(u)
        v_carray = CArray(v)
        if v_carray.shape[0] > 1:
            raise ValueError(
                "2nd array must have shape shape (1, n_features).")

        if v_carray.issparse is True:
            # Broadcasting not supported for sparse arrays
            v_broadcast = v_carray.repmat(u_carray.shape[0], 1)
        else:  # Broadcasting is supported by design for dense arrays
            v_broadcast = v_carray

        # Format of output array should be the same as v
        u_carray = u_carray.tosparse() if v_carray.issparse else u_carray.todense()

        diff = (u_carray - v_broadcast)

        k_grad = self._k(u_carray, v_carray)
        # Casting the kernel to sparse if needed for efficient broadcasting
        if diff.issparse is True:
            k_grad = k_grad.tosparse()

        return CArray(2 * self.gamma * diff * k_grad)
