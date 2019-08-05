"""
.. module:: CMetricAUC
   :synopsis: Performance Metric: Area Under (ROC) Curve

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
import sklearn.metrics as skm

from secml.array import CArray
from secml.ml.peval.metrics import CMetric
from secml.ml.peval.metrics import CRoc


class CMetricAUC(CMetric):
    """Performance evaluation metric: Area Under (ROC) Curve.

    AUC is computed using the trapezoidal rule.

    The metric uses:
     - y_true (true ground labels)
     - score (estimated target values)

    Attributes
    ----------
    class_type : 'auc'

    Notes
    -----
    This implementation is restricted to the binary classification task.

    Examples
    --------
    >>> from secml.ml.peval.metrics import CMetricAUC
    >>> from secml.array import CArray

    >>> peval = CMetricAUC()
    >>> print(peval.performance_score(CArray([0, 1, 0, 0]), score=CArray([0, 0, 0, 0])))
    0.5

    """
    __class_type = 'auc'
    best_value = 1.0

    def _performance_score(self, y_true, score):
        """Computes the Area Under the ROC Curve (AUC).

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
        fpr, tpr = CRoc().compute(y_true, score)[0:2]
        return float(skm.auc(fpr.tondarray(), tpr.tondarray()))
