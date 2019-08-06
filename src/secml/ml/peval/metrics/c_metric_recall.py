"""
.. module:: CMetricRecall
   :synopsis: Performance Metric: Recall

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
import sklearn.metrics as skm

from secml.array import CArray
from secml.ml.peval.metrics import CMetric


class CMetricRecall(CMetric):
    """Performance evaluation metric: Recall (True Positive Rate).

    The recall is the ratio tp / (tp + fn) where tp is the
    number of true positives and fn the number of false negatives.
    The recall is intuitively the ability of the classifier
    to find all the positive samples.
    This is equivalent to True Positive Rate.

    The metric uses:
     - y_true (true ground labels)
     - y_pred (predicted labels)

    Attributes
    ----------
    class_type : 'recall'

    Examples
    --------
    >>> from secml.ml.peval.metrics import CMetricRecall
    >>> from secml.array import CArray

    >>> peval = CMetricRecall()
    >>> print(peval.performance_score(CArray([0, 1, 2, 3]), CArray([0, 1, 1, 3])))
    0.75

    """
    __class_type = 'recall'
    best_value = 1.0

    def _performance_score(self, y_true, y_pred):
        """Computes the Recall score (True Positive Rate).

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
