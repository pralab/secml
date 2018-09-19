"""
.. module:: AttributesUtils
   :synopsis: Collection of utilities for attributes management

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from secml.core.type_utils import is_str

__all__ = ['as_public', 'as_private',
           'has_property', 'get_property', 'has_getter', 'has_setter',
           'is_public', 'is_protected', 'is_readonly', 'is_readwrite',
           'is_readable', 'is_writable', 'extract_attr']


def _check_is_attr_name(attr):
    """Raise TypeError if input is not an attribute name (string)."""
    if not is_str(attr):
        raise TypeError("attribute must be passed as a string.")


def as_public(attr):
    """Return the public name associated with a protected attribute.

    Examples
    --------
    >>> from secml.core.attr_utils import as_public

    >>> as_public('_attr1')
    'attr1'
    >>> as_public('attr1')  # Public attributes are returned as is
    'attr1'
    >>> as_public('__attr1')  # This is NOT a private attribute!
    '_attr1'

    """
    _check_is_attr_name(attr)
    import re
    return re.sub('^_rw_|^_r_|^_', '', attr)


def as_private(obj_class, attr):
    """Return the PRIVATE name associated with input attribute.

    Parameters
    ----------
    obj_class : any class
        Target class (usually extracted using obj.__class__).
    attr : str
        Name of the target attribute.

    """
    _check_is_attr_name(attr)
    return '_' + obj_class.__name__ + '__' + attr


def has_property(obj, attr):
    """True if attribute is a property or has an associated property.

    Parameters
    ----------
    obj : object
        Any class instance.
    attr : str
        Name of the attribute to check.

    """
    _check_is_attr_name(attr)
    return True if hasattr(obj.__class__, as_public(attr)) and \
                   isinstance(getattr(
                       obj.__class__, as_public(attr)), property) else False


def get_property(obj, attr):
    """Return the property associated with input attribute.

    If no property is associated with input attribute, raise AttributeError.

    Parameters
    ----------
    obj : object
        Any class instance.
    attr : str
        Name of the attribute to check.

    """
    _check_is_attr_name(attr)
    if not has_property(obj, attr):
        raise AttributeError("'{:}' has no property associated with attribute "
                             "'{:}'.".format(obj.__class__.__name__, attr))
    return getattr(obj.__class__, as_public(attr))


def has_getter(obj, attr):
    """True if an attribute has an associated getter.

    Parameters
    ----------
    obj : object
        Any class instance.
    attr : str
        Name of the attribute to check.

    """
    _check_is_attr_name(attr)
    return True if has_property(obj, attr) and \
                   get_property(obj, attr).fget is not None else False


def has_setter(obj, attr):
    """True if an attribute has an associated setter.

    Parameters
    ----------
    obj : object
        Any class instance.
    attr : str
        Name of the attribute to check.

    """
    _check_is_attr_name(attr)
    return True if has_property(obj, attr) and \
                   get_property(obj, attr).fset is not None else False


def is_public(obj, attr):
    """Return True if input attribute is PUBLIC.

    A public attribute has the name without '_' as a prefix.

    Parameters
    ----------
    obj : object
        Any class instance. --> NOT USED
    attr : str
        Name of the attribute to check.

    """
    _check_is_attr_name(attr)
    return True if not attr.startswith('_') else False


def is_readonly(obj, attr):
    """Return True if input attribute is READ ONLY.

    A read only attribute has ONLY a getter associated with it.

    Parameters
    ----------
    obj : object
        Any class instance.
    attr : str
        Name of the attribute to check.

    """
    _check_is_attr_name(attr)
    return True if not is_public(obj, attr) and has_getter(obj, attr) and \
                   not has_setter(obj, attr) else False


def is_readwrite(obj, attr):
    """Return True if input attribute is READ/WRITE.

    A read/write attribute has BOTH a getter AND a setter associated with it.

    Parameters
    ----------
    obj : object
        Any class instance.
    attr : str
        Name of the attribute to check.

    """
    _check_is_attr_name(attr)
    # There cannot be a setter without a getter!
    return True if has_setter(obj, attr) else False


def is_protected(obj, attr):
    """Return True if input attribute is PROTECTED.

    A protected attribute has the name starting with only '_'
    and no getter/setter associated with it.

    Parameters
    ----------
    obj : object
        Any class instance.
    attr : str
        Name of the attribute to check.

    """
    _check_is_attr_name(attr)
    # There cannot be a setter without a getter!
    return True if not is_public(obj, attr) and \
                   not has_getter(obj, attr) else False


def is_readable(obj, attr):
    """Return True if input attribute is READABLE.

    A readable attribute can be one of the following:
     - public
     - read/write (getter/setter associated with property)
     - read only (getter associated with property)

    Parameters
    ----------
    obj : object
        Any class instance.
    attr : str
        Name of the attribute to check.

    """
    _check_is_attr_name(attr)
    return True if is_public(obj, attr) or is_readwrite(obj, attr) or \
                   is_readonly(obj, attr) else False


def is_writable(obj, attr):
    """Return True if input attribute is WRITABLE.

    A writable attribute can be one of the following:
     - public
     - read/write (getter/setter associated with property)

    Parameters
    ----------
    attr : str
        Name of the attribute to check.

    """
    _check_is_attr_name(attr)
    return True if is_public(obj, attr) or is_readwrite(obj, attr) else False


def extract_attr(obj, mode):
    """Generates a sequence of attributes from an input dictionary.

    This function returns a generator with the dictionary's
    keys having a name compatible with specified mode.

    The following modalities are available:
     * 'pub' -> PUBLIC (no '_' in the prefix)
     * 'rw' -> READ/WRITE (a getter/setter is associated with it)
     * 'r' -> READ ONLY (a getter is associated with it)
     * 'pro' -> PROTECTED ('_' as the prefix and no getter/setter associated)

    All modes can be stacked up using '+' (see examples).

    Parameters
    ----------
    obj : any object
        Any class which attributes should be extracted.
    mode : str
        Extraction modality. All available modalities
        can be combined using a plus '+'.

    Notes
    -----
    Sorting of the attributes in the output generator is random.

    """

    def parse_modes(mode_str):
        """Parse modes string and return a list with the required checks."""
        mode_list = mode_str.split('+')
        req_check = []
        for m in mode_list:
            if m == 'pub':
                req_check.append(is_public)
            elif m == 'rw':
                req_check.append(is_readwrite)
            elif m == 'r':
                req_check.append(is_readonly)
            elif m == 'pro':
                req_check.append(is_protected)
            else:
                raise ValueError("mode `{:}` not supported.".format(m))
        return req_check

    # Parsing the modes string
    check_list = parse_modes(mode)
    # Yelding only the required attributes
    for attr in obj.__dict__:
        if any(e(obj, attr) for e in check_list):
            yield attr


import unittest
from secml.utils import CUnitTest


class TestAttributeUtilities(CUnitTest):
    """Unit test for secml.core.attr_utils."""

    def test_extract_attr(self):
        # Toy class for testing
        class Foo(object):
            def __init__(self):
                self.a = 5
                self._b = 5
                self._c = 5
                self._d = 5

            @property
            def b(self):
                pass

            @property
            def c(self):
                pass

            @c.setter
            def c(self):
                pass

        f = Foo()

        self.logger.info(
            "Testing attributes extraction based on accessibility...")

        # All cases... ugly but works :D
        self.assertTrue(set(attr for attr in extract_attr(f, 'pub')) == set(['a']))
        self.assertTrue(set(attr for attr in extract_attr(f, 'r')) == set(['_b']))
        self.assertTrue(set(attr for attr in extract_attr(f, 'rw')) == set(['_c']))
        self.assertTrue(set(attr for attr in extract_attr(f, 'pub+r')) == set(['a', '_b']))
        self.assertTrue(set(attr for attr in extract_attr(f, 'pub+rw')) == set(['a', '_c']))
        self.assertTrue(set(attr for attr in extract_attr(f, 'pub+pro')) == set(['a', '_d']))
        self.assertTrue(set(attr for attr in extract_attr(f, 'r+rw')) == set(['_b', '_c']))
        self.assertTrue(set(attr for attr in extract_attr(f, 'r+pro')) == set(['_b', '_d']))
        self.assertTrue(set(attr for attr in extract_attr(f, 'rw+pro')) == set(['_c', '_d']))
        self.assertTrue(set(attr for attr in extract_attr(f, 'pub+r+rw')) == set(['a', '_b', '_c']))
        self.assertTrue(set(attr for attr in extract_attr(f, 'pub+r+pro')) == set(['a', '_b', '_d']))
        self.assertTrue(set(attr for attr in extract_attr(f, 'pub+rw+pro')) == set(['a', '_c', '_d']))
        self.assertTrue(set(attr for attr in extract_attr(f, 'pub+r+rw+pro')) == set(['a', '_b', '_c', '_d']))


if __name__ == '__main__':
    unittest.main()
