"""
.. module:: OptimizationUtils
   :synopsis: Collection of mixed utilities for optimizers

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from secml.array import CArray

__all__ = ['fun_tondarray', 'fprime_tondarray' ]


def fun_tondarray(xk, fun, fprime, *args):
    """Wrapper for functions that work with CArrays.

    This function wraps any callable (bound) method that
    takes as input CArray(s) and return CArray(s) in order
    to take any array like object and return a np.ndarray.

    Even if both fun and fprime are required, this function
    runs `fun` only. See :meth:`_fprime_tondarray` for `fprime` wrapper.

    Parameters
    ----------
    xk : array_like
        First input of the function.
    fun : bound method
        Callable function that takes CArray(s) as input and
        return a CArray as result.
    fprime : bound method
        Callable function that takes CArray(s) as input and
        return a CArray as result.
    args : \*args, optional
        Extra arguments passed to `fun`.

    Returns
    -------
    out_fun : np.ndarray
        Output of 'fun' casted to ndarray.

    """
    # NOTE: fprime is not used. This is a wrapper for fun
    return CArray(fun(CArray(xk), *args)).tondarray()


def fprime_tondarray(xk, fun, fprime, *args):
    """Wrapper for functions that work with CArrays.

    This function wraps any callable (bound) method that
    takes as input CArray(s) and return CArray(s) in order
    to take any array like object and return a np.ndarray.

    Even if both fun and fprime are required, this function
    runs `fprime` only. See :meth:`_fun_tondarray` for `fun` wrapper.

    Parameters
    ----------
    xk : array_like
        First input of the function.
    fun : bound method
        Callable function that takes CArray(s) as input and
        return a CArray as result.
    fprime : bound method
        Callable function that takes CArray(s) as input and
        return a CArray as result.
    args : \*args, optional
        Extra arguments passed to `fun`.

    Returns
    -------
    out_fprime : np.ndarray
        Output of 'fun' casted to ndarray.

    """
    # NOTE: fun is not used. This is a wrapper for fprime
    return CArray(fprime(CArray(xk), *args)).tondarray()
