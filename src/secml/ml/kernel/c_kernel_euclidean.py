"""
.. module:: CKernelEuclidean
   :synopsis: Euclidean distances kernel.

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from sklearn import metrics

from secml.array import CArray
from secml.ml.kernel import CKernel


class CKernelEuclidean(CKernel):
    """Euclidean distances kernel.

    Given matrices X and Y, this is computed by::

        K(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))

    for each pair of rows in X and in Y.
    If parameter squared is True (default False), sqrt() operation is avoided.

    Attributes
    ----------
    class_type : 'euclidean'

    Parameters
    ----------
    batch_size : int or None, optional
        Size of the batch used for kernel computation. Default None.

    Examples
    --------
    >>> from secml.array import CArray
    >>> from secml.ml.kernel.c_kernel_euclidean import CKernelEuclidean

    >>> print(CKernelEuclidean().k(CArray([[1,2],[3,4]]), CArray([[10,20],[30,40]])))
    CArray([[20.124612 47.801674]
     [17.464249 45.      ]])

    >>> print(CKernelEuclidean().k(CArray([[1,2],[3,4]])))
    CArray([[0.       2.828427]
     [2.828427 0.      ]])

    """
    __class_type = 'euclidean'

    def _k(self, x, y, squared=False,
           x_norm_squared=None, y_norm_squared=None):
        """Compute the euclidean kernel between x and y.

        Parameters
        ----------
        x : CArray
            First array of shape (n_x, n_features).
        y : CArray
            Second array of shape (n_y, n_features).
        squared : bool, optional
            If True, return squared Euclidean distances. Default False
        x_norm_squared : CArray or None, optional
            Pre-computed dot-products of vectors in x (e.g., (x**2).sum(axis=1)).
        y_norm_squared : CArray or None, optional
            Pre-computed dot-products of vectors in y (e.g., (y**2).sum(axis=1)).

        Returns
        -------
        kernel : CArray
            Kernel between x and y, shape (n_x, n_y).

        See Also
        --------
        :meth:`.CKernel.k` : Main computation interface for kernels.

        """
        return CArray(metrics.pairwise.euclidean_distances(
            x.get_data(), y.get_data(), squared=squared,
            X_norm_squared=x_norm_squared, Y_norm_squared=y_norm_squared))

    def gradient(self, x, v, squared=False):
        """Calculates Euclidean distances kernel gradient wrt vector 'v'.

        The gradient of Euclidean distances kernel is given by::

            dK(x,v)/dv = - (x - v) / k(x,v)   if squared = False (default)
            dK(x,v)/dv = - 2 * (x - v)        if squared = True

        Parameters
        ----------
        x : CArray
            First array of shape (n_x, n_features).
        v : CArray
            Second array of shape (n_features, ) or (1, n_features).
        squared : bool, optional
            If True, return squared Euclidean distances. Default False

        Returns
        -------
        kernel_gradient : CArray
            Kernel gradient of x with respect to vector v. Array of
            shape (n_x, n_features) if n_x > 1, else a flattened
            array of shape (n_features, ).

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.ml.kernel.c_kernel_euclidean import CKernelEuclidean

        >>> array = CArray([[15,25],[45,55]])
        >>> vector = CArray([2,5])
        >>> print(CKernelEuclidean().gradient(array, vector))
        CArray([[-0.544988 -0.838444]
         [-0.652039 -0.758185]])

        >>> print(CKernelEuclidean().gradient(array, vector, squared=True))
        CArray([[ -26  -40]
         [ -86 -100]])

        >>> print(CKernelEuclidean().gradient(vector, vector))
        CArray([0. 0.])

        """
        # Checking if second array is a vector
        if v.is_vector_like is False:
            raise ValueError(
                "kernel gradient can be computed only wrt vector-like arrays.")

        x_2d = x.atleast_2d()
        v_2d = v.atleast_2d()

        grad = self._gradient(x_2d, v_2d, squared=squared)
        return grad.ravel() if x_2d.shape[0] == 1 else grad

    def _gradient(self, x, v, squared=False):
        """Calculates Euclidean distances kernel gradient wrt vector 'v'.

        The gradient of Euclidean distances kernel is given by::

            dK(x,v)/dv = - (x - v) / k(x,v)   if squared = False (default)
            dK(x,v)/dv = - 2 * (x - v)        if squared = True

        Parameters
        ----------
        x : CArray
            First array of shape (n_x, n_features).
        v : CArray
            Second array of shape (1, n_features).
        squared : bool, optional
            If True, return squared Euclidean distances. Default False

        Returns
        -------
        kernel_gradient : CArray
            Kernel gradient of u with respect to vector v,
            shape (1, n_features).

        See Also
        --------
        :meth:`.CKernel.gradient` : Gradient computation interface for kernels.

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

        if squared is True:  # - 2 * (x - y)
            return - 2 * diff

        k_grad = self._k(x, v)
        k_grad[k_grad == 0] = 1.0  # To avoid nans later

        # Casting the kernel to sparse if needed for efficient broadcasting
        if diff.issparse is True:
            k_grad = k_grad.tosparse()

        # - (x - y) / k(x - y)
        grad = - diff / k_grad
        # Casting to sparse if necessary
        return grad.tosparse() if diff.issparse else grad
