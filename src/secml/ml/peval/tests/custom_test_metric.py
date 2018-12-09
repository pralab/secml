from secml.ml.peval.metrics import CMetric
from secml.core.constants import nan


class CMetricFirstNan(CMetric):
    __class_type = 'some_nan'
    best_value = 1.0

    def _init_(self):
        self._count = 0

    def _performance_score(self, y_true, score):
        """Computes the True Positives ratio at given False Positives ratio.

        Parameters
        ----------
        y_true : CArray
            Flat array with true binary labels in range
            {0, 1} or {-1, 1} for each pattern.
        score : CArray
            Flat array with target scores for each pattern, can either be
            probability estimates of the positive class or confidence values.

        Warning: the result is equal to nan if only one element vectors are
        given

        Returns
        -------
        metric : float
            Returns metric value as float.

        Notes
        -----
        This implementation is restricted to the binary classification task.

        """
        if self._count == 0:
            self._count += 1
            return nan
        else:
            return 1

