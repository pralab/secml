"""
.. module:: MetricRecall
   :synopsis: Performance Metric: Recall

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
import sklearn.metrics as skm

from secml.array import CArray
from secml.peval.metrics import CMetric


class CMetricRecall(CMetric):
    """Performance evaluation metric: Recall.

    The recall is the ratio tp / (tp + fn) where tp is the
    number of true positives and fn the number of false negatives.
    The recall is intuitively the ability of the classifier
    to find all the positive samples.

    The metric uses:
     - y_true (true ground labels)
     - y_pred (predicted labels)

    Examples
    --------
    >>> from secml.peval.metrics import CMetricRecall
    >>> from secml.array import CArray

    >>> peval = CMetricRecall()
    >>> print peval.performance_score(CArray([0, 1, 2, 3]), CArray([0, 1, 1, 3]))
    0.75

    """
    class_type = 'recall'
    best_value = 1.0

    def _performance_score(self, y_true, y_pred):
        """Computes the Recall score.

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

        return float(skm.recall_score(
            y_true.tondarray(), y_pred.tondarray(), average=average))
