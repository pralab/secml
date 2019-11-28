"""
.. module:: AttributesUtils
   :synopsis: Collection of utilities for attributes management

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml import _NoValue
from secml.core.type_utils import is_str

__all__ = ['as_public',
           'as_protected', 'has_protected', 'get_protected',
           'as_private', 'has_private', 'get_private',
           'has_property', 'get_property', 'has_getter', 'has_setter',
           'add_readonly', 'add_readwrite',
           'is_public', 'is_protected', 'is_readonly', 'is_readwrite',
           'is_readable', 'is_writable', 'extract_attr']


def _check_is_attr_name(attr):
    """Raise TypeError if input is not an attribute name (string)."""
    if not is_str(attr):
        raise TypeError("attribute must be passed as a string, "
                        "not {:}.".format(type(attr)))


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


def as_protected(attr):
    """Return the protected name associated with a public attribute.

    Examples
    --------
    >>> from secml.core.attr_utils import as_protected

    >>> as_protected('attr1')
    '_attr1'
    >>> as_protected('__attr1')
    '_attr1'
    >>> as_protected('_attr1')  # Protected attributes are returned as is
    '_attr1'

    """
    _check_is_attr_name(attr)
    if not attr.startswith('_'):  # Public attribute
        return '_' + attr
    if attr.startswith('__'):  # Private attribute
        return attr[1:]  # Remove the first underscore
    return attr  # Already a protected attribute


def has_protected(obj, attr):
    """True if attribute is a protected attribute of class.

    Parameters
    ----------
    obj : object
        Target class instance.
    attr : str
        Name of the attribute to check.

    """
    return hasattr(obj, as_protected(attr))


def get_protected(obj_class, attr, default=_NoValue):
    """Return the protected attribute of class.

    Parameters
    ----------
    obj_class : class
        Target class (usually extracted using obj.__class__).
    attr : str
        Name of the attribute to return.
    default : any, optional
        Value that is returned when the named attribute is not found.

    """
    if default is not _NoValue:  # Pass default to getattr
        return getattr(obj_class, as_protected(attr), default)
    else:  # Standard getattr (error will be raise if attr is not found)
        return getattr(obj_class, as_protected(attr))


def as_private(obj_class, attr):
    """Return the PRIVATE name associated with input attribute.

    Parameters
    ----------
    obj_class : class
        Target class (usually extracted using obj.__class__).
    attr : str
        Name of the target attribute.

    """
    _check_is_attr_name(attr)
    attr = '__' + attr if attr.startswith('__') is False else attr
    return '_' + obj_class.__name__ + attr


def has_private(obj_class, attr):
    """True if attribute is a private attribute of class.

    Parameters
    ----------
    obj_class : class
        Target class (usually extracted using obj.__class__).
    attr : str
        Name of the attribute to check.

    """
    return hasattr(obj_class, as_private(obj_class, attr))


def get_private(obj_class, attr, default=_NoValue):
    """Return the private attribute of class.

    Parameters
    ----------
    obj_class : class
        Target class (usually extracted using obj.__class__).
    attr : str
        Name of the attribute to return.
    default : any, optional
        Value that is returned when the named attribute is not found.

    """
    if default is not _NoValue:  # Pass default to getattr
        return getattr(obj_class, as_private(obj_class, attr), default)
    else:  # Standard getattr (error will be raise if attr is not found)
        return getattr(obj_class, as_private(obj_class, attr))


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


def add_readonly(obj, attr, value=None):
    """Add a READ ONLY attribute to object.

    A read only attribute is defined as a protected attribute plus
    a getter associated with it.

    Parameters
    ----------
    obj : object
        Any class instance.
    attr : str
        Name of the attribute to set.
    value : any, optional
        Value to assign to the attribute. If not given, None is used.

    """
    if not has_protected(obj, attr):
        setattr(obj, as_protected(attr), value)

    def fget(get_obj):
        return getattr(get_obj, as_protected(attr))

    setattr(obj.__class__, attr, property(fget))


def add_readwrite(obj, attr, value=None):
    """Add a READ/WRITE attribute to object.

    A read/write attribute is defined as a protected attribute plus
    a getter AND a setter associated with it.

    Parameters
    ----------
    obj : object
        Any class instance.
    attr : str
        Name of the attribute to set.
    value : any, optional
        Value to assign to the attribute. If not given, None is used.

    """
    if not has_protected(obj.__class__, attr):
        setattr(obj, as_protected(attr), value)

    def fget(get_obj):
        return getattr(get_obj, as_protected(attr))

    def fset(set_obj, set_val):
        return setattr(set_obj, as_protected(attr), set_val)

    setattr(obj.__class__, attr, property(fget, fset))


def is_public(obj, attr):
    """Return True if input attribute is PUBLIC.

    A public attribute has the name without '_' as a prefix.

    Parameters
    ----------
    obj : object
        Any class instance.
    attr : str
        Name of the attribute to check.

    """
    _check_is_attr_name(attr)
    # Exclude properties to only return actual public attributes
    return True if not attr.startswith('_') and \
                   not has_property(obj, attr) else False


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
    return True if not is_public(obj, attr) and \
                   has_getter(obj, attr) and \
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
    return True if is_public(obj, attr) or \
                   is_readwrite(obj, attr) or \
                   is_readonly(obj, attr) else False


def is_writable(obj, attr):
    """Return True if input attribute is WRITABLE.

    A writable attribute can be one of the following:
     - public
     - read/write (getter/setter associated with property)

    Parameters
    ----------
    obj : object
        Any class instance.
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
     * 'pub' -> PUBLIC (standard attribute, no '_' in the prefix)
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
    # Yielding only the required attributes
    for attr in obj.__dict__:
        if any(e(obj, attr) for e in check_list):
            yield attr
