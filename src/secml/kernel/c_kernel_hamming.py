"""
.. module:: KernelHamming
   :synopsis: Hamming distances kernel

.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>
.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from sklearn import metrics

from prlib.array import CArray
from prlib.kernel import CKernel


class CKernelHamming(CKernel):
    """Hamming distance kernel.

    Attributes
    ----------
    usenumba : False as current kernel does not use Numba for optimization.

    Examples
    --------
    >>> from prlib.array import CArray
    >>> from prlib.kernel.c_kernel_hamming import CKernelHamming

    >>> print CKernelHamming().k(CArray([[1,2],[3,4]]), CArray([[10,20],[30,40]]))
    CArray([[ 1.  1.]
     [ 1.  1.]])

    >>> print CKernelHamming().k(CArray([[1,2],[3,4]]))
    CArray([[ 0.  1.]
     [ 1.  0.]])

    """
    class_type = 'hamming'
    usenumba = False

    def __init__(self, cache_size=100):

        super(CKernelHamming, self).__init__(cache_size=cache_size)

    def _k(self, x, y):
        """Compute the hamming distances kernel between x and y.

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
        return CArray(metrics.pairwise.pairwise_distances(
            CArray(x).get_data(), CArray(y).get_data(), metric='hamming'))

    def _gradient(self, u, v):
        """Calculate hamming kernel gradient wrt vector 'v'."""
        raise NotImplementedError(
            "Gradient of Hamming Kernel is not available.")
