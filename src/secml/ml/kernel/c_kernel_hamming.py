"""
.. module:: CKernelHamming
   :synopsis: Hamming distances kernel

.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from sklearn import metrics

from secml.array import CArray
from secml.ml.kernel import CKernel


class CKernelHamming(CKernel):
    """Hamming distance kernel.

    Attributes
    ----------
    class_type : 'hamming'

    Parameters
    ----------
    batch_size : int or None, optional
        Size of the batch used for kernel computation. Default None.

    Examples
    --------
    >>> from secml.array import CArray
    >>> from secml.ml.kernel.c_kernel_hamming import CKernelHamming

    >>> print(CKernelHamming().k(CArray([[1,2],[3,4]]), CArray([[10,20],[30,40]])))
    CArray([[1. 1.]
     [1. 1.]])

    >>> print(CKernelHamming().k(CArray([[1,2],[3,4]])))
    CArray([[0. 1.]
     [1. 0.]])

    """
    __class_type = 'hamming'

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
