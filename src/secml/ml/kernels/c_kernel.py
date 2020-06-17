"""
.. module:: Kernel
   :synopsis: Interface for kernel functions

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Angelo Sotgiu <angelo.sotgiu@unica.it>

"""
from abc import ABCMeta

from secml.ml import CModule
from secml.array import CArray
from secml.core.decorators import deprecated


class CKernel(CModule, metaclass=ABCMeta):
    """Abstract class that defines basic methods for kernels.

    A kernel is a pairwise metric that compute the distance
    between sets of patterns.

    Kernels can be considered similarity measures,
    i.e. s(a, b) > s(a, c) if objects a and b are considered
    "more similar" than objects a and c.
    A kernel must be positive semi-definite (PSD), even though non-PSD kernels
    can also be used to train classifiers (e.g., SVMs, but losing convexity).

    Parameters
    ----------
    preprocess : CModule or None, optional
        Features preprocess to be applied to input data.
        Can be a CModule subclass. If None, input data is used as is.

    """
    __super__ = 'CKernel'

    def __init__(self, preprocess=None):
        self._rv = None
        self._cached_kernel = None
        super(CKernel, self).__init__(preprocess=preprocess)

    def _clear_cache(self):
        """Clears cached values within this class instance."""
        self._cached_kernel = None
        super(CKernel, self)._clear_cache()

    def _check_is_fitted(self):
        pass  # TODO: check rv

    def _fit(self, x, y=None):
        self.rv = x
        return self

    @property
    def rv(self):
        """Reference vectors with respect to compute the kernel."""
        return self._rv

    @rv.setter
    def rv(self, rv):
        """Sets the reference vectors with respect to the kernel will be
        computed.

        Parameters
        ----------
        rv : CArray
            One or more reference vectors.

        """
        self._rv = CArray(rv).atleast_2d()

    def k(self, x, rv=None):
        """Compute kernel between x and rv.

        Parameters
        ----------
        x : CArray
            First array of shape (n_x, n_features).
        rv : CArray, optional
            Second array of shape (n_rv, n_features). If not specified,
            it is set to x and the kernel k(x,x) is computed.

        Returns
        -------
        kernel : CArray or scalar
            Kernel between x and rv.
            Array of shape (n_x, n_rv) or scalar if both x and y are
            vector-like.

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.ml.kernels import CKernelRBF

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
        CArray([[1.]])

        """
        # apply pre-processing (if any) on rv
        rv = x.atleast_2d() if rv is None else rv.atleast_2d()
        if self.preprocess is not None:
            rv = self._forward_preprocess(rv, caching=False)
        # store preprocessed rv within the class
        self.rv = rv

        kernel = self.forward(x, caching=False)

        # If both x and rv are vectors, return scalar
        if x.is_vector_like and self._rv.is_vector_like:
            return kernel.item()
        else:
            return kernel
