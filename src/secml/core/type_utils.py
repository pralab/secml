"""
.. module:: TypeUtils
   :synopsis: Collection of utility functions for types management

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
import numpy as np
from scipy.sparse import issparse

__all__ = ['is_bool', 'is_int', 'is_intlike', 'is_float', 'is_floatlike',
           'is_scalar', 'is_scalarlike', 'is_list', 'is_list_of_lists',
           'is_ndarray', 'is_scsarray', 'is_slice', 'is_str', 'is_tuple',
           'is_set', 'is_dict', 'to_builtin']


def is_bool(x):
    return isinstance(x, (bool, np.bool_))


def is_int(x):
    if isinstance(x, int) and not isinstance(x, bool):  # bool subclass of int
        return True
    elif isinstance(x, (long, np.integer)):
        return True
    return False


def is_intlike(x):
    """Return True if input is integer or list/array of 1 integer.

    Examples
    --------
    >>> from secml.core.type_utils import is_intlike

    >>> print is_intlike(0)  # Standard int
    True
    >>> print is_intlike(0.1)  # Standard float
    False

    >>> print is_intlike(np.array([0]))  # ndarray with one int
    True
    >>> print is_intlike(np.array([0.1]))  # ndarray with one float
    False

    """

    if is_int(x):
        return True  # built-in or numpy integers
    elif (is_list(x) and len(x) == 1 and is_int(x[0])) or \
            (is_ndarray(x) and x.size == 1 and x.dtype.kind in ('i', 'u')):
        return True
    else:
        return False


def is_float(x):
    return isinstance(x, (float, np.floating))


def is_floatlike(x):
    """Return True if input is float or list/array of 1 float.

    Examples
    --------
    >>> from secml.core.type_utils import is_floatlike

    >>> print is_floatlike(0.1)  # Standard float
    True
    >>> print is_floatlike(0)  # Standard int
    False

    >>> print is_floatlike(np.array([0.1]))  # ndarray with one float
    True
    >>> print is_floatlike(np.array([0]))  # ndarray with one int
    False

    """
    if is_float(x):
        return True  # built-in or numpy floats
    elif (is_list(x) and len(x) == 1 and is_float(x[0])) or \
            (is_ndarray(x) and x.size == 1 and x.dtype.kind in ('f')):
        return True
    else:
        return False


def is_scalar(x):
    """Return True if input is integer or float."""
    return is_int(x) or is_float(x)


def is_scalarlike(x):
    """Return True if input is scalar (int or float) or list/array of 1 real."""
    return is_intlike(x) or is_floatlike(x)


def is_list(x):
    return isinstance(x, list)


def is_list_of_lists(x):
    """Return True if input is a list of lists, otherwise False.

    Examples
    --------
    >>> is_list_of_lists([[1, 2], [3]])
    True
    >>> is_list_of_lists([[1], 2, [3]])
    False
    >>> is_list_of_lists([])
    False

    """
    if not is_list(x):  # Not a list
        return False
    elif len(x) == 0:  # Empty list
        return False
    elif any(not is_list(elem) for elem in x):  # One or more elems not lists
        return False
    return True


def is_ndarray(x):
    return isinstance(x, np.ndarray)


def is_scsarray(x):
    """Returns True if input is a scipy.sparse array."""
    return issparse(x)


def is_slice(x):
    return isinstance(x, slice)


def is_str(x):
    # FIXME: CHANGE THIS FOR PYTHON 3 (STR ARE UNICODE)
    return isinstance(x, (str, np.str_, unicode))


def is_tuple(x):
    return isinstance(x, tuple)


def is_set(x):
    return isinstance(x, set)


def is_dict(x):
    return isinstance(x, dict)


def to_builtin(x):
    """Convert input to the corresponding built-in type.

    Works with the following types:
     - bool, np.bool_ -> bool
     - int, long, np.integer -> int
     - float, np.floating -> float
     - str, np.str_ -> str

    """
    if is_bool(x):
        # Covers bool, np.bool_
        return bool(x)
    elif is_int(x):
        # Covers int, long, np.integer
        return int(x)
    elif is_float(x):
        # Covers float, np.floating
        return float(x)
    elif is_str(x):
        # Covers str, np.str_
        return str(x)
    else:
        raise TypeError("objects of type {:} not supported.".format(type(x)))
