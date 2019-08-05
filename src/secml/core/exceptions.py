"""
.. module:: Exceptions
   :synopsis: Custom errors and warnings

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""

__all__ = ['NotFittedError']


class NotFittedError(ValueError, AttributeError):
    """Exception to raise if the object is used before training.

    This class inherits from both ValueError and AttributeError.

    Examples
    --------
    >>> from secml.ml.classifiers import CClassifierSVM
    >>> from secml.array import CArray
    >>> from secml.core.exceptions import NotFittedError
    >>> try:
    ...     CClassifierSVM().predict(CArray([[1, 2]]))
    ... except NotFittedError as e:
    ...     print(repr(e))
    ...                        # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    NotFittedError('this `CClassifierSVM` is not trained. Call `.fit()` first.',)

    """
    pass
