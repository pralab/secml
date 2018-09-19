"""
.. module:: SquaredLossNumba
   :synopsis: Squared Loss Function \w Numba Optimization

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from secml.classifiers.loss import CLoss
from numba import jit


class CLossSquaredNumba(CLoss):
    """Squared Loss Function \w Numba Optimization

    Ordinary least squares fit.

    Notes
    -----
    Numba optimized loss functions works with scalar.
    For array-compatible functions see content of
    `.secml.classifiers.loss` package.

    """

    class_type = 'squared_loss'
    loss_type = 'regression'
    usenumba = True

    @staticmethod
    @jit(['float32(int32, float32)', 'float64(int64, float64)'], nopython=True)
    def loss(y_true, y_pred):
        """Compute Squared Loss.

        Parameters
        ----------
        y_true : int
            True label of current sample. Must be -1/+1.
        y_pred : float
            Predicted label of current sample.

        """
        return 0.5 * ((y_pred - y_true) ** 2)

    @staticmethod
    @jit(['float32(int32, float32)', 'float64(int64, float64)'], nopython=True)
    def dloss(y_true, y_pred):
        """Compute Squared Loss Derivative.

        Parameters
        ----------
        y_true : int
            True label of current sample. Must be -1/+1.
        y_pred : float
            Predicted label of current sample.

        """
        return y_pred - y_true
