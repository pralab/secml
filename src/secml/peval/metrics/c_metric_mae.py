"""
.. module:: MetricMAE
   :synopsis: Performance Metric: Mean Absolute Error

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
import sklearn.metrics as skm

from secml.array import CArray
from secml.peval.metrics import CMetric


class CMetricMAE(CMetric):
    """Performance evaluation metric: Mean Absolute Error.

    Regression loss of ground truth (correct labels) and
    the predicted regression score.

    The metric uses:
     - y_true (true ground labels)
     - score (estimated target values)

    Examples
    --------
    >>> from secml.peval.metrics import CMetricMAE
    >>> from secml.array import CArray

    >>> peval = CMetricMAE()
    >>> print peval.performance_score(CArray([0, 1, 0, 0]), score=CArray([0, 0, 0, 0]))
    0.25

    """
    class_type = 'mae'
    best_value = 0.0

    def _performance_score(self, y_true, score):
        """Computes the Mean Absolute Error.

        Parameters
        ----------
        y_true : CArray
            Ground truth (true) labels or target scores.
        score : CArray
            Estimated target values.

        Returns
        -------
        metric : float
            Returns metric value as float.

        """
        return float(skm.mean_absolute_error(y_true.tondarray(),
                                             score.tondarray()))
