import warnings
import functools

__all__ = ["deprecated"]


class deprecated:
    """Decorator to mark a function or class as deprecated.

    Issue a warning when the function is called/the class is instantiated and
    adds a warning to the docstring.

    The optional extra argument will be appended to the deprecation message
    and the docstring.

    Note: to use this with the default value for extra,
    put in an empty of parentheses:
    >>> from secml.core.decorators import deprecated
    >>> deprecated() # doctest: +ELLIPSIS
    <secml.core.decorators.deprecated object at ...>

    >>> @deprecated()
    ... def some_function(): pass

    Parameters
    ----------
    version : str
        Version since which the function or class is deprecated.
    extra : str, optional
        Extra text to be added to the deprecation messages.

    Notes
    -----
    Adapted from:
     - https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/deprecation.py
     - https://wiki.python.org/moin/PythonDecoratorLibrary

    """
    def __init__(self, version, extra=''):
        self.extra = extra
        self.version = version

    def __call__(self, obj):
        """Call method.

        Parameters
        ----------
        obj : class or function
            The object to decorate. Can be a class or a function.

        """
        if isinstance(obj, type):
            return self._decorate_class(obj)
        else:
            return self._decorate_fun(obj)

    def _decorate_class(self, cls):
        """Decorate class clf."""
        msg = "class `{:}` is deprecated since version {:}".format(
            cls.__name__, self.version)
        if self.extra:
            msg += "; %s" % self.extra

        warnings.filterwarnings(
            'once', message=msg, category=DeprecationWarning)

        # FIXME: we should probably reset __new__ for full generality
        init = cls.__init__

        def wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return init(*args, **kwargs)
        cls.__init__ = wrapped

        wrapped.__name__ = '__init__'
        wrapped.deprecated_original = init

        cls.__doc__ = self._update_doc(cls.__doc__)

        return cls

    def _decorate_fun(self, fun):
        """Decorate function fun."""
        msg = "function `{:}` is deprecated since version {:}".format(
            fun.__name__, self.version)
        if self.extra:
            msg += "; %s" % self.extra

        warnings.filterwarnings(
            'once', message=msg, category=DeprecationWarning)

        @functools.wraps(fun)
        def wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return fun(*args, **kwargs)

        wrapped.__doc__ = self._update_doc(wrapped.__doc__)
        # Add a reference to the wrapped function so that we can introspect
        # on function arguments in Python 2 (already works in Python 3)
        wrapped.__wrapped__ = fun

        return wrapped

    def _update_doc(self, olddoc):
        """Update the docstring of the class/function adding
        'Deprecated since version XX' + the extra optional text."""
        newdoc = ".. deprecated:: {:}".format(self.version)
        if self.extra:
            newdoc = "%s\n   %s" % (newdoc, self.extra)
        if olddoc:
            newdoc = "%s\n\n%s" % (newdoc, olddoc)
        else:  # A docstring, even empty, is required for correct visualization
            newdoc = "%s\n%s" % (newdoc, '""""""')
        return newdoc
