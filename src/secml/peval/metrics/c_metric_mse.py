"""
.. module:: MetricMSE
   :synopsis: Performance Metric: Mean Squared Error

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
import sklearn.metrics as skm

from prlib.array import CArray
from prlib.peval.metrics import CMetric


class CMetricMSE(CMetric):
    """Performance evaluation metric: Mean Squared Error.

    Regression loss of ground truth (correct labels) and
    the predicted regression score.

    The metric uses:
     - y_true (true ground labels)
     - score (estimated target values)

    Examples
    --------
    >>> from prlib.peval.metrics import CMetricMSE
    >>> from prlib.array import CArray

    >>> peval = CMetricMSE()
    >>> print peval.performance_score(CArray([0, 1, 0, 0]), score=CArray([0, 0, 0, 0]))
    0.25

    """
    class_type = 'mse'
    best_value = 0.0

    def _performance_score(self, y_true, score):
        """Computes the Mean Squared Error.

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
        return float(skm.mean_squared_error(y_true.tondarray(),
                                            score.tondarray()))
