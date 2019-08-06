"""
.. module:: CMetricTestError
   :synopsis: Performance Metric: Test Error

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
import sklearn.metrics as skm

from secml.array import CArray
from secml.ml.peval.metrics import CMetric


class CMetricTestError(CMetric):
    """Performance evaluation metric: Test Error.

    Test Error score is the percentage (inside 0/1 range)
    of wrongly predicted labels (inverse of accuracy).

    The metric uses:
     - y_true (true ground labels)
     - y_pred (predicted labels)

    Attributes
    ----------
    class_type : 'test-error'

    Examples
    --------
    >>> from secml.ml.peval.metrics import CMetricTestError
    >>> from secml.array import CArray

    >>> peval = CMetricTestError()
    >>> print(peval.performance_score(CArray([0, 1, 2, 3]), CArray([0, 1, 1, 3])))
    0.25

    """
    __class_type = 'test-error'
    best_value = 0.0

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
        return 1.0 - float(skm.accuracy_score(y_true.tondarray(),
                                              y_pred.tondarray()))
