"""
.. module:: KernelInterface
   :synopsis: Interface for Kernel metrics

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from abc import ABCMeta, abstractmethod
import six
from six.moves import range

import warnings

from secml.core import CCreator
from secml.array import CArray


@six.add_metaclass(ABCMeta)
class CKernel(CCreator):
    """Abstract class that defines basic methods for kernels.

    A kernel is a pairwise metric that compute the distance
    between sets of patterns.

    Kernels can be considered similarity measures,
    i.e. s(a, b) > s(a, c) if objects a and b are considered
    "more similar" than objects a and c.
    A kernel must also be positive semi-definite.

    Parameters
    ----------
    batch_size : int or None, optional
        Size of the batch used for kernel computation. Default None.

        .. deprecated:: 0.10

    """
    __super__ = 'CKernel'

    def __init__(self, batch_size=None):

        if batch_size is not None:
            warnings.warn(
                "`batch_size`'` is deprecated since version 0.10.",
                DeprecationWarning)
        self.batch_size = batch_size

    @abstractmethod
    def _k(self, x, y):
        """Private method that computes kernel.

        .. warning::

            This method must be reimplemented by subclasses.

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

        """
        raise NotImplementedError()

    def k(self, x, y=None):
        """Compute kernel between x and y.

        Parameters
        ----------
        x : CArray
            First array of shape (n_x, n_features).
        y : CArray, optional
            Second array of shape (n_y, n_features). If not specified,
            the kernel k(x,x) is computed.

        Returns
        -------
        kernel : CArray or scalar
            Kernel between x and y. Array of shape (n_x, n_y) or scalar
            if both x and y are vector-like.

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.ml.kernel import CKernelRBF

        >>> array1 = CArray([[15,25],[45,55]])
        >>> array2 = CArray([[10,20],[40,50]])
        >>> print(CKernelRBF().k(array1, array2))
        CArray([[1.92875e-22 0.00000e+00]
         [0.00000e+00 1.92875e-22]])

        >>> print(CKernelRBF().k(array1))
        CArray([[1. 0.]
         [0. 1.]])

        >>> vector = CArray([15,25])
        >>> print(CKernelRBF().k(vector, array1))
        CArray([[1. 0.]])
        >>> print(CKernelRBF().k(array1, vector))
        CArray([[1.]
         [0.]])
        >>> print(CKernelRBF().k(vector, vector))
        1.0

        """
        y = x if y is None else y  # If y is not specified we compute k(x,x)

        # Converting separately to 2D as we need original shape later
        x_carray_2d = x.atleast_2d()
        y_carray_2d = y.atleast_2d()

        kernel = CArray(self._k(x_carray_2d, y_carray_2d)).todense()

        # If both x and y are vectors, return scalar
        if x.is_vector_like and y.is_vector_like:
            return kernel.item()
        else:
            return kernel

    def similarity(self, x, y=None):
        """Computes kernel. Wrapper of 'k' function.

        See Also
        --------
        :meth:`.CKernel.k` : Main computation interface for kernels.

        """
        return self.k(x, y)

    @abstractmethod
    def _gradient(self, u, v):
        """Private method that computes kernel gradient wrt to 'v'.

        .. warning::

            This method must be reimplemented by subclasses.

        Parameters
        ----------
        u : CArray or array_like
            First array of shape (1, n_features).
        v : CArray or array_like
            Second array of shape (1, n_features).

        Returns
        -------
        kernel_gradient : CArray
            Kernel gradient of u with respect to vector v. Array of
            shape (1, n_features)

        """
        raise NotImplementedError()

    def gradient(self, x, v):
        """Calculates kernel gradient wrt vector 'v'.

        Parameters
        ----------
        x : CArray
            First array of shape (n_x, n_features).
        v : CArray
            Second array of shape (n_features, ) or (1, n_features).

        Returns
        -------
        kernel_gradient : CArray
            Kernel gradient of u with respect to vector v. Array of
            shape (n_x, n_features) if n_x > 1, else a flattened
            array of shape (n_features, ).

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.ml.kernel import CKernelRBF

        >>> array = CArray([[15,25],[45,55]])
        >>> vector = CArray([2,5])
        >>> print(CKernelRBF(gamma=1e-4).gradient(array, vector))
        CArray([[0.002456 0.003779]
         [0.005567 0.006473]])

        >>> print(CKernelRBF().gradient(vector, vector))
        CArray([0. 0.])

        """
        # Recasting data for safety... cost-free for any CArray
        x_carray = x.atleast_2d()
        v_carray = v.atleast_2d()

        # Checking if second array is a vector
        if v_carray.ndim > 1 and v_carray.shape[0] > 1:
            raise ValueError(
                "kernel gradient can be computed only wrt vector-like arrays.")

        # Instancing an empty array to avoid return errors
        grad = CArray([], tosparse=x_carray.issparse)
        # Kernel gradient can be dense or sparse depending on `x_carray`
        for i in range(x_carray.shape[0]):
            grad_i = self._gradient(x_carray[i, :], v_carray)
            grad = grad_i if i == 0 else grad.append(grad_i, axis=0)

        # grad = self._gradient(x_carray, v_carray)

        return grad.ravel() if x_carray.shape[0] == 1 else grad
