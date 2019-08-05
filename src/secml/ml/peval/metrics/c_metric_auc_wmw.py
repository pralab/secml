"""
.. module:: CMetricAUCWMW
   :synopsis: Performance Metric: Area Under (ROC) Curve using Wilcoxon-Mann-Whitney statistic

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml.array import CArray
from secml.ml.peval.metrics import CMetric


class CMetricAUCWMW(CMetric):
    """Performance evaluation metric: Area Under (ROC) Curve with Wilcoxon-Mann-Whitney statistic.

    The metric uses:
     - y_true (true ground labels)
     - score (estimated target values)

    Attributes
    ----------
    class_type : 'auc-wmw'

    Notes
    -----
    This implementation is restricted to the binary classification task.

    Examples
    --------
    >>> from secml.ml.peval.metrics import CMetricAUCWMW
    >>> from secml.array import CArray

    >>> peval = CMetricAUCWMW()
    >>> print(peval.performance_score(CArray([0, 1, 0, 0]), score=CArray([0, 0, 0, 0])))
    0.5

    """
    __class_type = 'auc-wmw'
    best_value = 1.0

    def _performance_score(self, y_true, score):
        """Computes the Area Under the ROC Curve (AUC) using the Wilcoxon-Mann-Whitney statistic.

        Parameters
        ----------
        y_true : CArray
            Flat array with true binary labels in range {0, 1}.
        score : CArray
            Flat array with target scores for each pattern, can either be
            probability estimates of the positive class or confidence values.

        Returns
        -------
        metric : float
            Returns metric value as float.

        Notes
        -----
        This implementation is restricted to the binary classification task
        with labels in range {0, 1}.

        """
        if CArray(CArray(y_true != 0).logical_and(y_true != 1)).any():
            raise ValueError("input labels should be binary in 0/1 interval.")

        idxp = y_true.find(y_true == 1)
        idxn = y_true.find(y_true == 0)

        auc = 0.0
        for i in idxp:
            for j in idxn:
                if score[i] > score[j]:
                    auc += 1.0
                elif score[i] == score[j]:
                    auc += 0.5

        return auc / (len(idxp) * len(idxn))
