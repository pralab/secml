"""
.. module:: KernelLaplacian
   :synopsis: Laplacian kernel

.. moduleauthor:: Paolo Russu <paolo.russu@diee.unica.it>


"""
from sklearn import metrics

from secml.array import CArray
from secml.ml.kernel import CKernel
import numpy as np


class CKernelLaplacian(CKernel):
    """Laplacian Kernel.

    Given matrices X and Y, this is computed by::

        K(x, y) = exp(-gamma |x-y|)

    for each pair of rows in X and in Y.

    Attributes
    ----------
    cache_size : int, size of the cache used for kernel computation. Default 100.

    Parameters
    ----------
    gamma : float
        Default is 1.0. Equals to `-0.5 * sigma^-2` in the standard
        formulation of rbf kernel, it is a free parameter to be used
        for balancing.

    Examples
    --------
    >>> from secml.array import CArray
    >>> from secml.ml.kernel.c_kernel_laplacian import CKernelLaplacian

    >>> print CKernelLaplacian(gamma=0.01).k(CArray([[1,2],[3,4]]), CArray([[10,20],[30,40]]))
    CArray([[ 0.76337949  0.51170858]
     [ 0.7945336   0.5325918 ]])

    >>> print CKernelLaplacian().k(CArray([[1,2],[3,4]]))
    CArray([[ 1.          0.01831564]
     [ 0.01831564  1.        ]])

    """
    class_type = 'laplacian'

    def __init__(self, gamma=1.0, cache_size=100):
        # Calling CKernel constructor
        super(CKernelLaplacian, self).__init__(cache_size=cache_size)
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
            laplacian kernel, is a free parameter to be used
            to balance the computed metric.

        """
        self._gamma = float(gamma)

    def _k(self, x, y):
        """Compute the laplacian kernel between x and y.

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
        :meth:`.CKernel.k` : Main computation interface for kernels.

        """
        # sklearn > 0.17 has the class pairwise.laplacian_kernel
        # For compatibility reasons, we keep using pairwise_distances
        K = metrics.pairwise.pairwise_distances(
            CArray(x).get_data(), CArray(y).get_data(), metric='cityblock')
        return CArray(np.exp(-self.gamma * K))

    def _gradient(self, u, v):
        """Calculate Laplacian kernel gradient wrt vector 'v'.

        The gradient of Laplacian kernel is given by::

            dK(u,v)/dv =  gamma * k(u,v) * sign(u - v)

        Parameters
        ----------
        u : CArray
            First array of shape (1, n_features).
        v : CArray
            Second array of shape (1, n_features).

        Returns
        -------
        kernel_gradient : CArray
            Kernel gradient of u with respect to vector v,
            shape (1, n_features).

        See Also
        --------
        :meth:`.CKernel.gradient` : Gradient computation interface for kernels.

        """
        u_carray = CArray(u)
        v_carray = CArray(v)
        if u_carray.shape[0] + v_carray.shape[0] > 2:
            raise ValueError(
                "Both input arrays must be 2-Dim of shape (1, n_features).")

        return CArray(self.gamma *
                      self._k(u_carray, v_carray) *
                      (u_carray - v_carray).sign())
