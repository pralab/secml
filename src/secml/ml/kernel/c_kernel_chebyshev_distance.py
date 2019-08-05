"""
.. module:: CKernelChebyshevDistance
   :synopsis: Chebyshev distances kernel

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>


"""
from sklearn import metrics

from secml.array import CArray
from secml.ml.kernel import CKernel


class CKernelChebyshevDistance(CKernel):
    """Chebyshev distances kernel.

    Given matrices X and Y, this is computed as::

        K(x, y) = max(|x - y|)

    for each pair of rows in X and in Y.

    Attributes
    ----------
    class_type : 'chebyshev-dist'

    Parameters
    ----------
    batch_size : int or None, optional
        Size of the batch used for kernel computation. Default None.

    Examples
    --------
    >>> from secml.array import CArray
    >>> from secml.ml.kernel.c_kernel_chebyshev_distance import CKernelChebyshevDistance

    >>> print(CKernelChebyshevDistance().k(CArray([[1,2],[3,4]]), CArray([[5,6],[7,8]])))
    CArray([[4. 6.]
     [2. 4.]])

    >>> print(CKernelChebyshevDistance().k(CArray([[1,2],[3,4]])))
    CArray([[0. 2.]
     [2. 0.]])

    """
    __class_type = 'chebyshev-dist'

    def __init__(self, gamma=1.0, batch_size=None):

        super(CKernelChebyshevDistance, self).__init__(batch_size=batch_size)

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
        """Compute the Chebyshev distances kernel between x and y.

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
        return CArray(metrics.pairwise.pairwise_distances(
            x.get_data(), y.get_data(), metric='chebyshev'))

    def _gradient(self, u, v):
        """Calculate Chebyshev distances kernel gradient wrt vector 'v'.

        The gradient of Chebyshev distances kernel is given by::

            dK(u,v)/dv =  k(u,v) * sign(u - v)

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
        :meth:`CKernel.gradient` : Gradient computation interface for kernels.

        """
        u_carray = CArray(u)
        v_carray = CArray(v)
        if u_carray.shape[0] + v_carray.shape[0] > 2:
            raise ValueError(
                "Both input arrays must be 2-Dim of shape (1, n_features).")

        g = u - v
        m = abs(g).max()
        g[abs(g) != m] = 0
        g[g == m] = 1
        g[g == -m] = -1

        return self._k(u_carray, v_carray) * g
