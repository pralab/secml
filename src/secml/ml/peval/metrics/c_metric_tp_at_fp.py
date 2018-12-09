"""
.. module:: MetricTPatFP
   :synopsis: Performance Metric: True Positives @ False Positives Ratio

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from secml.array import CArray
from secml.ml.peval.metrics import CMetric
from secml.ml.peval.metrics import CRoc


class CMetricTPatFP(CMetric):
    """Performance evaluation metric: True Positives @ False Positives Ratio.

    The metric uses:
     - y_true (true ground labels)
     - score (estimated target values)

    Attributes
    ----------
    fp_rate : float
        Desired False Positives rate in the interval [0,1]. Default 0.01 (1%)

    Notes
    -----
    This implementation is restricted to the binary classification task.

    Examples
    --------
    >>> from secml.ml.peval.metrics import CMetricTPatFP
    >>> from secml.array import CArray

    >>> peval = CMetricTPatFP(fp_rate=0.5)
    >>> print peval.performance_score(CArray([0, 1, 0, 0]), score=CArray([0, 0, 0, 0]))
    0.5

    """
    __class_type = 'tp_at_fp'
    best_value = 1.0

    def __init__(self, fp_rate=0.01):

        # False positives rate @ which true positives should be computed
        self.fp_rate = float(fp_rate)

    def _performance_score(self, y_true, score):
        """Computes the True Positives ratio at given False Positives ratio.

        Parameters
        ----------
        y_true : CArray
            Flat array with true binary labels in range
            {0, 1} or {-1, 1} for each pattern.
        score : CArray
            Flat array with target scores for each pattern, can either be
            probability estimates of the positive class or confidence values.

        Returns
        -------
        metric : float
            Returns metric value as float.

        Warning
        -------
        The result is equal to nan if only one element vectors are given.

        Notes
        -----
        This implementation is restricted to the binary classification task.

        """
        return CArray(self.fp_rate).interp(
            *CRoc().compute(y_true, score)[0:2]).ravel()
