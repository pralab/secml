"""
.. module:: FunctionUtils
   :synopsis: Collection of mixed utility functions

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""

__all__ = ['OrderedFlexibleClass']


class OrderedFlexibleClass(object):
    """A flexible class exposing its attributes in a specific order when iterated.

    Order of the attributes inside the class follows the inputs sequence.
    Any attribute set after class initialization will be placed at the end
    of attributes sequence (see examples).

    Parameters
    ----------
    items : tuple1, tuple2, ...
        Any custom sequence of tuples with the attributes to set.
        Each tuple must be a (key, value) pair.

    Examples
    --------
    >>> from prlib.utils import OrderedFlexibleClass

    >>> c = OrderedFlexibleClass(('attr1', None), ('attr2', 5))
    >>> print tuple(attr for attr in c)
    (None, 5)

    >>> c.attr3 = 123
    >>> print tuple(attr for attr in c)
    (None, 5, 123)

    """

    def __init__(self, *items):
        if len(items) == 0:
            raise ValueError("class must have at least one attribute.")
        if not all(isinstance(i, tuple) for i in items):
            raise TypeError("each attribute must be specified as a tuple of (key, value).")
        # List with attributes sequence (this provides the fixed order)
        self._params = []
        # __setattr__ will store the attribute in `_params` and set its value
        for i in items:
            setattr(self, *i)

    @property
    def attr_order(self):
        """Returns a list specifing current attributes order."""
        return self._params

    def __setattr__(self, key, value):
        """Set desired attribute and store the key in `_params`."""
        # Register attribute only if new (skip service attribute _params)
        if key != '_params' and not hasattr(self, key):
            self._params.append(key)
        # Set attribute value in the standard way
        super(OrderedFlexibleClass, self).__setattr__(key, value)

    def __iter__(self):
        """Returns class attributes following a fixed order."""
        for e in self._params:
            yield self.__dict__[e]
