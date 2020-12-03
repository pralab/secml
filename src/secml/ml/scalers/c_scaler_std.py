"""
.. module:: CScalerStd
   :synopsis: Standard Scaler.

.. moduleauthor:: Marco Meloni <m.meloni42@studenti.unica.it>

"""
from sklearn.preprocessing import StandardScaler

from secml.ml.scalers import CScalerSkLearn
from secml.array import CArray


class CScalerStd(CScalerSkLearn):
    """CScalerStd.

    Parameters
    ----------
    copy : boolean, optional, default True
        If False, try to avoid a copy and do inplace scaling instead.
    with_mean : boolean, True by default
        If True, center the data before scaling. This does not work (and will
        raise an exception) when attempted on sparse matrices, because
        centering them entails building a dense matrix which in common use
        cases is likely to be too large to fit in memory.
    with_std : boolean, True by default
        If True, scale the data to unit variance (or equivalently, unit
        standard deviation).
    Attributes
    ----------
    class_type : 'std'

    """

    __class_type = 'std'

    def __init__(self, copy=True, with_mean=True, with_std=True,
                 preprocess=None):
        scaler = StandardScaler(
            copy=copy, with_mean=with_mean, with_std=with_std)

        super(CScalerStd, self).__init__(
            sklearn_scaler=scaler, preprocess=preprocess)

    def _check_is_fitted(self):
        """Check if the scaler is trained (fitted).

        Raises
        ------
        NotFittedError
            If the scaler is not fitted.

        """
        self._check_is_fitted_scaler(self, ['n_samples_seen_'])

    def _backward(self, w=None):
        self._check_is_fitted()

        v = CArray(self.sklearn_scaler.scale_).deepcopy()
        v[v != 0] = 1 / v[v != 0]  # avoids division by zero

        return w * v if w is not None else v
