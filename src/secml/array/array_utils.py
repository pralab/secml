"""
.. module:: ArrayUtils
   :synopsis: Collection of utility functions for CArray and subclasses

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
import numpy as np
from scipy.sparse import issparse

from secml.core.type_utils import is_int, is_bool, is_tuple, is_slice

__all__ = ['is_vector_index', 'tuple_atomic_tolist', 'tuple_sequence_tondarray']


def is_vector_index(idx):
    """Check if input index is valid for vector-like arrays.

    An array is vector-like when 1-Dimensional or
    2-Dimensional with shape[0] == 1.

    Parameters
    ----------
    idx : int, bool, slice
        Index to check.

    Returns
    -------
    out_check : bool
        Return True if input is a valid index for
        any axis with size 1, else False.

    """
    return True if (np.asanyarray(idx) == 0 or np.asanyarray(idx) == -1 or  # integers 0, -1
                    (np.asanyarray(idx) == True and np.asanyarray(idx).dtype in (bool, np.bool_)) or  # True but not '1'
                    idx == slice(None, None, None) or  # :
                    idx == slice(0, None, None) or  # 0:
                    idx == slice(0, 1, None) or  # 0:1
                    idx == slice(None, 1, None) or  # :1
                    idx == slice(-1, 0, None)  # -1
                    ) else False


def tuple_atomic_tolist(idx):
    """Convert tuple atomic elements to list.

    Atomic objects converted:
        - `int`, `np.integer`
        - `bool`, `np.bool_`

    Parameters
    ----------
    idx : tuple
        Tuple which elements have to be converted.

    Returns
    -------
    out_tuple : tuple
        Converted tuple.

    """
    if not is_tuple(idx):
        raise TypeError("input must be a tuple")
    return tuple([[elem] if is_int(elem) or is_bool(elem) else elem for elem in idx])


def tuple_sequence_tondarray(idx):
    """Convert sequences inside tuple to ndarray.

    A sequence can be:
        - int, `np.integer`
        - bool, `np.bool_`
        - list
        - `np.ndarray`
        - CDense
        - CSparse (are converted to dense first)
        - CArray

    Parameters
    ----------
    idx : tuple
        Tuple which elements have to be converted.

    Returns
    -------
    out_tuple : tuple
        Converted tuple.

    """
    if not is_tuple(idx):
        raise TypeError("input must be a tuple")
    # Converting CArray/CSparse/CDense to ndarray
    idx = tuple([elem.tondarray() if
                 hasattr(elem, 'tondarray') else elem for elem in idx])
    # Converting not-slice and not-None to ndarray
    return tuple([np.asarray(elem) if not (is_slice(elem) or elem is None)
                  else elem for elem in idx])
