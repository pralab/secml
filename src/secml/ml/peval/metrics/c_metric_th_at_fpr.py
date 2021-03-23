"""
.. module:: MetricTHatFPR
   :synopsis: ROC Threshold @ False Positive Rate

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml.array import CArray
from secml.ml.peval.metrics import CMetric
from secml.ml.peval.metrics import CRoc


class CMetricTHatFPR(CMetric):
    """Performance evaluation metric: ROC Threshold @ False Positive Rate.

    The metric uses:
     - y_true (true ground labels)
     - score (estimated target values)

    Parameters
    ----------
    fpr : float
        Desired False Positive Rate in the interval [0,1]. Default 0.01 (1%)

    Attributes
    ----------
    class_type : 'th-at-fpr'

    Notes
    -----
    This implementation is restricted to the binary classification task.

    Examples
    --------
    >>> from secml.ml.peval.metrics import CMetricTHatFPR
    >>> from secml.array import CArray

    >>> peval = CMetricTHatFPR(fpr=0.5)
    >>> peval.performance_score(CArray([0, 1, 0, 0]), score=CArray([0, 0, 0, 0]))
    0.0005

    """
    __class_type = 'th-at-fpr'
    best_value = 1.0

    def __init__(self, fpr=0.01):
        self.fpr = float(fpr)

    def _performance_score(self, y_true, score):
        """Computes the ROC Threshold at given False Positive Rate.

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

        Notes
        -----
        This implementation is restricted to the binary classification task.

        """
        fp, tp, th = CRoc().compute(y_true, score)
        return CArray(self.fpr).interp(fp, th).item()
