"""
.. module:: CMetricMSE
   :synopsis: Performance Metric: Mean Squared Error

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
import sklearn.metrics as skm

from secml.array import CArray
from secml.ml.peval.metrics import CMetric


class CMetricMSE(CMetric):
    """Performance evaluation metric: Mean Squared Error.

    Regression loss of ground truth (correct labels) and
    the predicted regression score.

    The metric uses:
     - y_true (true ground labels)
     - score (estimated target values)

    Attributes
    ----------
    class_type : 'mse'

    Examples
    --------
    >>> from secml.ml.peval.metrics import CMetricMSE
    >>> from secml.array import CArray

    >>> peval = CMetricMSE()
    >>> print(peval.performance_score(CArray([0, 1, 0, 0]), score=CArray([0, 0, 0, 0])))
    0.25

    """
    __class_type = 'mse'
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
