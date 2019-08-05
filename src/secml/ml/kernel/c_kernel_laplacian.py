"""
.. module:: CKernelLaplacian
   :synopsis: Laplacian kernel

.. moduleauthor:: Marco Melis <marco.melis@unica.it>


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
    batch_size : int or None, optional
        Size of the batch used for kernel computation. Default None.

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

    def __init__(self, gamma=1.0, batch_size=None):

        super(CKernelLaplacian, self).__init__(batch_size=batch_size)

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

        The gradient of laplacian kernel is given by::

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

    def gradient(self, x, v):
        """Calculates laplacian kernel gradient wrt vector 'v'.

        The gradient of laplacian kernel is given by::

            dK(x,v)/dv =  gamma * k(x,v) * sign(x - v)

        Parameters
        ----------
        x : CArray
            First array of shape (n_x, n_features).
        v : CArray
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
        >>> from secml.ml.kernel.c_kernel_laplacian import CKernelLaplacian

        >>> array = CArray([[15,0], [0,55]])
        >>> vector = CArray([2,5])
        >>> print(CKernelLaplacian(gamma=0.01).gradient(array, vector))
        CArray([[ 0.008353 -0.008353]
         [-0.005945  0.005945]])

        >>> print(CKernelLaplacian().gradient(vector, vector))
        CArray([0. 0.])

        """
        # Checking if second array is a vector
        if v.is_vector_like is False:
            raise ValueError(
                "kernel gradient can be computed only wrt vector-like arrays.")

        x_2d = x.atleast_2d()
        v_2d = v.atleast_2d()

        grad = self._gradient(x_2d, v_2d)
        return grad.ravel() if x_2d.shape[0] == 1 else grad

    def _gradient(self, x, v):
        """Calculate laplacian kernel gradient wrt vector 'v'.

        The gradient of laplacian kernel is given by::

            dK(x,v)/dv =  gamma * k(x,v) * sign(x - v)

        Parameters
        ----------
        x : CArray
            First array of shape (n_x, n_features).
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
