"""
.. module:: TypeUtils
   :synopsis: Collection of utility functions for types management

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
import numpy as np
from scipy.sparse import issparse

__all__ = ['is_bool', 'is_int', 'is_intlike', 'is_float', 'is_floatlike',
           'is_scalar', 'is_scalarlike', 'is_inf', 'is_posinf', 'is_neginf',
           'is_nan', 'is_list', 'is_list_of_lists',
           'is_ndarray', 'is_scsarray', 'is_slice', 'is_str', 'is_bytes',
           'is_tuple', 'is_set', 'is_dict', 'to_builtin']


def is_bool(x):
    return isinstance(x, (bool, np.bool_))


def is_int(x):
    if isinstance(x, int) and not isinstance(x, bool):  # bool is a subclass of int
        return True
    elif isinstance(x, np.integer):
        return True
    return False


def is_intlike(x):
    """Return True if input is integer or list/array of 1 integer.

    Examples
    --------
    >>> from secml.core.type_utils import is_intlike

    >>> print(is_intlike(0))  # Standard int
    True
    >>> print(is_intlike(0.1))  # Standard float
    False

    >>> print(is_intlike(np.array([0])))  # ndarray with one int
    True
    >>> print(is_intlike(np.array([0.1])))  # ndarray with one float
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

    >>> print(is_floatlike(0.1))  # Standard float
    True
    >>> print(is_floatlike(0))  # Standard int
    False

    >>> print(is_floatlike(np.array([0.1])))  # ndarray with one float
    True
    >>> print(is_floatlike(np.array([0])))  # ndarray with one int
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
    """True if input is integer or float."""
    return is_int(x) or is_float(x)


def is_scalarlike(x):
    """True if input is scalar (int or float) or list/array of 1 real."""
    return is_intlike(x) or is_floatlike(x)


def is_inf(x):
    """True if input is a positive/negative infinity.

    Parameters
    ----------
    x : scalar

    Examples
    --------
    >>> from secml.core.type_utils import is_inf
    >>> from secml.core.constants import inf, nan

    >>> print(is_inf(inf))
    True
    >>> print(is_inf(-inf))
    True

    >>> print(is_inf(nan))
    False

    >>> print(is_inf(0.1))
    False

    >>> from secml.array import CArray
    >>> print(is_inf(CArray([inf])))  # Use `CArray.is_inf()` instead
    Traceback (most recent call last):
        ...
    TypeError: input must be a scalar.

    """
    if not is_scalar(x):
        raise TypeError("input must be a scalar.")
    return np.isinf(x)


def is_posinf(x):
    """True if input is a positive infinity.

    Parameters
    ----------
    x : scalar

    Examples
    --------
    >>> from secml.core.type_utils import is_posinf
    >>> from secml.core.constants import inf, nan

    >>> print(is_posinf(inf))
    True

    >>> print(is_posinf(-inf))
    False

    >>> from secml.array import CArray
    >>> print(is_posinf(CArray([inf])))  # Use `CArray.is_posinf()` instead
    Traceback (most recent call last):
        ...
    TypeError: input must be a scalar.

    """
    if not is_scalar(x):
        raise TypeError("input must be a scalar.")
    return np.isposinf(x)


def is_neginf(x):
    """True if input is a negative infinity.

    Parameters
    ----------
    x : scalar

    Examples
    --------
    >>> from secml.core.type_utils import is_neginf
    >>> from secml.core.constants import inf, nan

    >>> print(is_neginf(-inf))
    True

    >>> print(is_neginf(inf))
    False

    >>> from secml.array import CArray
    >>> print(is_neginf(CArray([-inf])))  # Use `CArray.is_neginf()` instead
    Traceback (most recent call last):
        ...
    TypeError: input must be a scalar.

    """
    if not is_scalar(x):
        raise TypeError("input must be a scalar.")
    return np.isneginf(x)


def is_nan(x):
    """True if input is Not a Number (NaN).

    Parameters
    ----------
    x : scalar

    Notes
    -----
    NumPy uses the IEEE Standard for Binary Floating-Point for
    Arithmetic (IEEE 754). This means that Not a Number is not
    equivalent to infinity.

    Examples
    --------
    >>> from secml.core.type_utils import is_nan
    >>> from secml.core.constants import inf, nan

    >>> print(is_nan(nan))
    True

    >>> print(is_nan(inf))
    False

    >>> print(is_nan(0.1))
    False

    >>> from secml.array import CArray
    >>> print(is_neginf(CArray([nan])))  # Use `CArray.is_nan()` instead
    Traceback (most recent call last):
        ...
    TypeError: input must be a scalar.

    """
    if not is_scalar(x):
        raise TypeError("input must be a scalar.")
    return np.isnan(x)


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


def is_str(x):  # text unicode strings
    if isinstance(x, str):
        return True
    elif isinstance(x, (np.str_, np.unicode_)):
        return True
    return False


def is_bytes(x):  # byte strings
    if isinstance(x, (bytes, np.bytes_)):
        return True
    return False


def is_tuple(x):
    return isinstance(x, tuple)


def is_set(x):
    return isinstance(x, set)


def is_dict(x):
    return isinstance(x, dict)


def to_builtin(x):
    """Convert input to the corresponding built-in type.

    Works with the following types:
     - `bool`, `np.bool_` -> `bool`
     - `int`, `np.integer` -> `int`
     - `float, `np.floating` -> `float`
     - `str`, `np.str_`, `np.unicode_` -> `str`
     - `bytes`, `np.bytes_` -> `bytes`

    """
    if is_bool(x):
        # Covers bool, np.bool_
        return bool(x)
    elif is_int(x):
        # Covers int, np.integer
        return int(x)
    elif is_float(x):
        # Covers float, np.floating
        return float(x)
    elif is_str(x):
        # Covers str, np.str_, np.unicode_
        return str(x)
    elif is_bytes(x):
        # Covers bytes, np.bytes_
        return bytes(x)
    else:
        raise TypeError("objects of type {:} not supported.".format(type(x)))
