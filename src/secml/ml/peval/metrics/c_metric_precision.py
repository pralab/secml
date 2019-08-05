"""
.. module:: CMetricPrecision
   :synopsis: Performance Metric: Precision

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
import sklearn.metrics as skm

from secml.array import CArray
from secml.ml.peval.metrics import CMetric


class CMetricPrecision(CMetric):
    """Performance evaluation metric: Precision.

    The precision is the ratio tp / (tp + fp) where tp is the
    number of true positives and fp the number of false positives.
    The precision is intuitively the ability of the classifier
    not to label as positive a sample that is negative.

    The metric uses:
     - y_true (true ground labels)
     - y_pred (predicted labels)

    Attributes
    ----------
    class_type : 'precision'

    Examples
    --------
    >>> from secml.ml.peval.metrics import CMetricPrecision
    >>> from secml.array import CArray

    >>> peval = CMetricPrecision()
    >>> print(peval.performance_score(CArray([0, 1, 2, 3]), CArray([0, 1, 1, 3])))
    0.625

    """
    __class_type = 'precision'
    best_value = 1.0

    def _performance_score(self, y_true, y_pred):
        """Computes the Precision score.

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

        return float(skm.precision_score(
            y_true.tondarray(), y_pred.tondarray(), average=average))
