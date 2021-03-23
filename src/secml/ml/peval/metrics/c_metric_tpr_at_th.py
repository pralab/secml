"""
.. module:: MetricTPRatTH
   :synopsis: Performance Metric: True Positive Rate @ ROC Threshold

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml.array import CArray
from secml.ml.peval.metrics import CMetric
from secml.core.type_utils import is_list


class CMetricTPRatTH(CMetric):
    """Performance evaluation metric: True Positive Rate @ ROC Threshold.

    The metric uses:
     - y_true (true ground labels)
     - score (estimated target values)

    Parameters
    ----------
    th : float or list
        ROC Threshold to use for computing True Positive Rate. Default 0.
        This can be a list of multiple values.

    Attributes
    ----------
    class_type : 'tpr-at-th'

    Examples
    --------
    >>> from secml.ml.peval.metrics import CMetricTPRatTH
    >>> from secml.array import CArray

    >>> peval = CMetricTPRatTH(th=1.7)
    >>> peval.performance_score(CArray([1, 1, 0, 0]), score=CArray([1.6, 2, 0.5, -1]))
    0.5

    """
    __class_type = 'tpr-at-th'
    best_value = 1.0

    def __init__(self, th=0.0):
        self.th = float(th) if is_list(th) is False else th

    def _performance_score(self, y_true, score, rep_idx=0):
        """Computes the True Positive Rate @ ROC Threshold.

        Parameters
        ----------
        y_true : CArray
            Ground truth (true) labels or target scores.
        score : CArray
            Flat array with target scores for each pattern, can either be
            probability estimates of the positive class or confidence values.
        rep_idx : int, optional
            Index of the th value to use. Default 0.

        Returns
        -------
        metric : float
            Returns metric value as float.

        """
        th = self.th[rep_idx] if is_list(self.th) is True else self.th
        p = CArray(y_true == 1)  # Positives
        return float(CArray(score[p] - th >= 0).sum()) / p.sum()
