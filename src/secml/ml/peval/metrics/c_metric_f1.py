"""
.. module:: CMetricF1
   :synopsis: Performance Metric: F1

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
import sklearn.metrics as skm

from secml.array import CArray
from secml.ml.peval.metrics import CMetric


class CMetricF1(CMetric):
    """Performance evaluation metric: F1.

    The F1 score can be interpreted as a weighted average
    of the precision and recall, where an F1 score reaches
    its best value at 1 and worst score at 0.

    The relative contribution of precision and recall to
    the F1 score are equal. The formula for the F1 score is::

        F1 = 2 * (precision * recall) / (precision + recall)

    The metric uses:
     - y_true (true ground labels)
     - y_pred (predicted labels)

    Attributes
    ----------
    class_type : 'f1'

    Examples
    --------
    >>> from secml.ml.peval.metrics import CMetricF1
    >>> from secml.array import CArray

    >>> peval = CMetricF1()
    >>> print(peval.performance_score(CArray([0, 1, 2, 3]), CArray([0, 1, 1, 3])))
    0.6666666666666666

    """
    __class_type = 'f1'
    best_value = 1.0

    def _performance_score(self, y_true, y_pred):
        """Computes the F1 score.

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
        if y_true.unique().size > 2:  # Multiclass data
            average = 'weighted'
        else:  # Default case
            average = 'binary'

        return float(skm.f1_score(
            y_true.tondarray(), y_pred.tondarray(), average=average))
