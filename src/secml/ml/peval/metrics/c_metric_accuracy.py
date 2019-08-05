"""
.. module:: CMetricAccuracy
   :synopsis: Performance Metric: Accuracy

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
import sklearn.metrics as skm

from secml.array import CArray
from secml.ml.peval.metrics import CMetric


class CMetricAccuracy(CMetric):
    """Performance evaluation metric: Accuracy.

    Accuracy score is the percentage (inside 0/1 range)
    of correctly predicted labels.

    The metric uses:
     - y_true (true ground labels)
     - y_pred (predicted labels)

    Attributes
    ----------
    class_type : 'accuracy'

    Examples
    --------
    >>> from secml.ml.peval.metrics import CMetricAccuracy
    >>> from secml.array import CArray

    >>> peval = CMetricAccuracy()
    >>> print(peval.performance_score(CArray([0, 1, 2, 3]), CArray([0, 1, 1, 3])))
    0.75

    """
    __class_type = 'accuracy'
    best_value = 1.0

    def _performance_score(self, y_true, y_pred):
        """Computes the Accuracy score.

        Parameters
        ----------
        y_true : CArray
            Ground truth (true) labels or target scores.
        y_pred : CArray
            Predicted labels, as returned by a CClassifier.

        Returns
        -------
        metric : float
            Returns metric value as float.

        """
        return float(skm.accuracy_score(y_true.tondarray(),
                                        y_pred.tondarray()))
