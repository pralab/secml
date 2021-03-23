"""
.. module:: CScalerMinMax
   :synopsis: Range Scaler.

.. moduleauthor:: Marco Meloni <m.meloni42@studenti.unica.it>

"""
from sklearn.preprocessing import MinMaxScaler

from secml.ml.scalers import CScalerSkLearn
from secml.array import CArray


class CScalerMinMax(CScalerSkLearn):
    """CScalerMinMax.

    Parameters
    ----------
    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.
    copy : bool, default=True
        Set to False to perform inplace row normalization and avoid a copy.

    Attributes
    ----------
    class_type : 'minmax'

    """

    __class_type = 'minmax'

    def __init__(self, feature_range=(0, 1), copy=True, preprocess=None):
        scaler = MinMaxScaler(feature_range=feature_range, copy=copy)

        super(CScalerMinMax, self).__init__(
            sklearn_scaler=scaler, preprocess=preprocess)

    def _check_is_fitted(self):
        """Check if the scaler is trained (fitted).

        Raises
        ------
        NotFittedError
            If the scaler is not fitted.

        """
        self._check_is_fitted_scaler(self, ['min_', 'n_samples_seen_'])

    def _backward(self, w=None):
        self._check_is_fitted()

        v = CArray(self.sklearn_scaler.data_range_).deepcopy()
        v[v != 0] = 1 / v[v != 0]  # avoids division by zero

        return w * v if w is not None else v
