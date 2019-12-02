"""
.. module:: FunctionUtils
   :synopsis: Collection of mixed utility classes and functions

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""

__all__ = ['AverageMeter', 'OrderedFlexibleClass', 'check_is_fitted']


class AverageMeter:
    """Computes and stores the average and current value.

    Attributes
    ----------
    val : float
        Current value.
    avg : float
        Average.
    sum : float
        Cumulative sum of seen values.
    count : int
        Number of seen values.

    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        """Updated average and current value.

        Parameters
        ----------
        val : float
            New current value.
        n : int, optional
            Multiplier for the current value. Indicates how many times
            the value should be counted in the average. Default 1.

        """
        val = float(val)
        n = int(n)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class OrderedFlexibleClass:
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
    >>> from secml.utils import OrderedFlexibleClass

    >>> c = OrderedFlexibleClass(('attr1', None), ('attr2', 5))
    >>> print(tuple(attr for attr in c))
    (None, 5)

    >>> c.attr3 = 123
    >>> print(tuple(attr for attr in c))
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


def check_is_fitted(obj, attributes, msg=None, check_all=True):
    """Check if the input object is trained (fitted).

    Checks if the input object is fitted by verifying if all or any of the
    input attributes are not None.

    Parameters
    ----------
    obj : object
        Instance of the class to check. Must implement `.fit()` method.
    attributes : str or list of str
        Attribute or list of attributes to check.
        Es.: `['classes', 'n_features', ...], 'classes'`
    msg : str or None, optional
        If None, the default error message is:
        "this `{name}` is not trained. Call `.fit()` first.".
        For custom messages if '{name}' is present in the message string,
        it is substituted by the class name of the checked object.
    check_all : bool, optional
        Specify whether to check (True) if all of the given attributes
        are not None or (False) just any of them. Default True.

    Raises
    ------
    NotFittedError
        If `check_all` is True and any of the attributes is None;
        if `check_all` is False and all of attributes are None.

    """
    from secml.core.type_utils import is_list, is_str
    from secml.core.exceptions import NotFittedError

    if msg is None:
        msg = "this `{name}` is not trained. Call `.fit()` first."

    if not hasattr(obj, 'fit'):
        raise TypeError("`{:}` does not implement `.fit()`.".format(obj))

    if is_str(attributes):
        attributes = [attributes]
    elif not is_list(attributes):
        raise TypeError(
            "the attribute(s) to check must be a string or a list of strings")

    condition = any if check_all is True else all

    if condition([getattr(obj, attr) is None for attr in attributes]):
        raise NotFittedError(msg.format(name=obj.__class__.__name__))
