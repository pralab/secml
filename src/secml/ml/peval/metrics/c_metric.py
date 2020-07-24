"""
.. module:: CMetric
   :synopsis: Interface for for performance evaluation metrics.

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from abc import ABCMeta, abstractmethod
import inspect

from secml.core import CCreator


class CMetric(CCreator, metaclass=ABCMeta):
    """Performance evaluation metrics.

    Utility functions to measure classification performance.
    Some metrics might require probability estimates of the positive class,
    confidence values, or binary decisions values.

    Each metric can be use either y_true (true ground labels) or
    y_pred (predicted labels) or score (predicted scores) or
    other data as inputs. Check documentation of each metric
    for more information.

    Attributes
    ----------
    best_value : best metric value. This is commonly a scalar (0.0 or 1.0).

    Examples
    --------
    >>> from secml.ml.peval.metrics import CMetric
    >>> from secml.array import CArray

    >>> peval = CMetric.create('accuracy')
    >>> print(peval.performance_score(y_true=CArray([0, 1, 2, 3]), y_pred=CArray([0, 1, 1, 3])))
    0.75

    >>> peval = CMetric.create('tpr-at-fpr', fpr=0.1)
    >>> print(peval.performance_score(y_true=CArray([0, 1, 0, 0]), score=CArray([1, 1, 0, 0])))
    0.3

    """
    __super__ = 'CMetric'

    best_value = None

    def performance_score(
            self, y_true=None, y_pred=None, score=None, **kwargs):
        """Compute the performance metric.

        Each metric can use as input either:
         - y_true (true ground labels)
         - y_pred (predicted labels)
         - score (predicted scores)
         - or any other data

        Check documentation of each metric for more information.

        If not all the required data is passed, TypeError will be raised.

        """
        # Add y_true, y_pred, score (the common parameters)
        kwargs.update(y_true=y_true, y_pred=y_pred, score=score)

        # Getting specifications of _performance_score method of the metric
        getargspec = inspect.getfullargspec
        metric_argspec = getargspec(self._performance_score)
        metric_params = metric_argspec.args[1:]  # Excluding `self`
        metric_defaults = metric_argspec.defaults

        # Check if all required parameters have been passed
        # Do not raise error if a defaulted parameter is not passed
        for p_idx, p in enumerate(metric_params):
            if kwargs.get(p, None) is None and \
                    (metric_defaults is None or
                     len(metric_params) - len(metric_defaults) > p_idx):
                raise TypeError("metric '{:}' requires '{:}' parameter".format(
                    self.class_type, p))

        # Clean any other kwarg passed and not required by the metric
        for p in list(kwargs):
            if p not in metric_params:
                kwargs.pop(p)

        # Call the metric and return the score
        return self._performance_score(**kwargs)

    @abstractmethod
    def _performance_score(self, *args, **kwargs):
        """Compute the performance metric.

        This must be reimplemented by subclasses.

        """
        raise NotImplementedError()
