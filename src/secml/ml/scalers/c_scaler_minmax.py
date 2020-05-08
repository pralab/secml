"""
.. module:: CScalerMinMax
   :synopsis: Range Scaler.

.. moduleauthor:: Marco Meloni <m.meloni42@studenti.unica.it>

"""
from sklearn.preprocessing import MinMaxScaler
from secml.ml.scalers.c_scaler_sklearn import CScalerSkLearn


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
        scaler = MinMaxScaler(feature_range, copy)

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
        grad = CScalerSkLearn._grad_calc(self.sklearn_scaler.data_range_,
                                         self._grad_funct)
        return w * grad if w is not None else grad

    def _grad_funct(self, x):
        return 0 if x == 0 else 1 / x
