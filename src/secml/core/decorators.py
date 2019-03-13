import warnings
import functools

__all__ = ["deprecated"]


class deprecated(object):
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
    extra : str
      Extra text to be added to the deprecation messages.

    Notes
    -----
    Adapted from:
     - https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/deprecation.py
     - https://wiki.python.org/moin/PythonDecoratorLibrary

    """
    def __init__(self, extra=''):
        self.extra = extra

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
        msg = "class `%s` is deprecated" % cls.__name__
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
        wrapped.__doc__ = self._update_doc(init.__doc__)
        wrapped.deprecated_original = init

        return cls

    def _decorate_fun(self, fun):
        """Decorate function fun."""
        msg = "function `%s` is deprecated" % fun.__name__
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
        'DEPRECATED' + the extra optional text."""
        newdoc = "DEPRECATED"
        if self.extra:
            newdoc = "%s: %s" % (newdoc, self.extra)
        if olddoc:
            newdoc = "%s\n\n%s" % (newdoc, olddoc)
        return newdoc
