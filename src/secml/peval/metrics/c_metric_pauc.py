"""
.. module:: MetricPartialAUC
   :synopsis: Performance Metric: Partial Area Under (ROC) Curve

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
import sklearn.metrics as skm

from secml.array import CArray
from secml.peval.metrics import CMetric
from secml.peval.metrics import CRoc


class CMetricPartialAUC(CMetric):
    """Performance evaluation metric: Partial Area Under (ROC) Curve.

    ROC is only considered between 0 and `fp_rate` false positives.

    AUC is computed using the trapezoidal rule.

    The metric uses:
     - y_true (true ground labels)
     - score (estimated target values)

    Attributes
    ----------
    fp_rate : float
        Desired false positives rate in the interval [0,1]. Default 0.01 (1%)
    n_points : int
        Number of points to be used when interpolating the partial ROC.
        Higher points means more accurate values but slower computation.
        Default 1000.

    Notes
    -----
    This implementation is restricted to the binary classification task.

    Examples
    --------
    >>> from secml.peval.metrics import CMetricPartialAUC
    >>> from secml.array import CArray

    >>> peval = CMetricPartialAUC(fp_rate=0.5)
    >>> print peval.performance_score(CArray([0, 1, 0, 0]), score=CArray([0, 0, 0, 0]))
    0.125

    """
    class_type = 'pauc'
    best_value = 1.0

    def __init__(self, fp_rate=0.01, n_points=1000):

        # False positives rate @ which true positives should be computed
        self.fp_rate = float(fp_rate)
        # Number of points to be used when interpolating ROC
        self.n_points = int(n_points)

    def _performance_score(self, y_true, score):
        """Computes the Partial Area Under the ROC Curve (AUC).

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
        fp_roc, tp_roc = CRoc().compute(y_true, score)[0:2]
        # Interpolating the ROC between 0 and fp_rate FP
        # Considering a number of points proportional to what used inside CRoc
        fpr = CArray.linspace(0, self.fp_rate, self.n_points)
        tpr = fpr.interp(fp_roc, tp_roc)

        return skm.auc(fpr.tondarray(), tpr.tondarray())
