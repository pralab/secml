"""
.. module:: HingeLossNumba
   :synopsis: Hinge Loss Function \w Numba Optimization

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from prlib.classifiers.loss import CLoss
from numba import jit


class CLossHingeNumba(CLoss):
    """Hinge Loss Function (soft-margin) \w Numba Optimization.

    Useful to construct Support Vector Machines.

    Notes
    -----
    Numba optimized loss functions works with scalar.
    For array-compatible functions see content of
    `.prlib.classifiers.loss` package.

    """
    
    class_type = 'hinge'
    loss_type = 'classification'
    usenumba = True

    @staticmethod
    @jit(['float32(int32, float32)', 'float64(int64, float64)'], nopython=True)
    def loss(y, score):
        """Compute Hinge Loss.

        Parameters
        ----------
        y : int
            True label of current sample. Must be -1/+1
        score : float
            Classification score of current sample.

        """
        h = 1 - y * score
        if h <= 0:
            return 0.0
        return h

    @staticmethod
    @jit(['float32(int32, float32)', 'float64(int64, float64)'], nopython=True)
    def dloss(y, score):
        """Compute Hinge Loss Derivative.

        Parameters
        ----------
        y : int
            True label of current sample. Must be -1/+1
        score : float
            Classification score of current sample.

        """
        z = y * score
        if z <= 1:
            return -y
        return 0.0


class CLossSquaredHingeNumba(CLoss):
    """Squared Hinge Loss Function (soft-margin) \w Numba Optimization.

    Useful to construct Support Vector Machines.

    Notes
    -----
    Numba optimized loss functions works with scalar.
    For array-compatible functions see content of
    `.prlib.classifiers.loss` package.

    """

    loss_type = 'squared_hinge'
    usenumba = True

    @staticmethod
    @jit(['float32(int32, float32)', 'float64(int64, float64)'], nopython=True)
    def loss(y, score):
        """Compute Squared Hinge Loss.

        Parameters
        ----------
        y : int
            True label of current sample. Must be -1/+1
        score : float
            Classification score of current sample.

        """
        z = 1 - y * score
        if z > 0:
            return z * z
        return 0.0

    @staticmethod
    @jit(['float32(int32, float32)', 'float64(int64, float64)'], nopython=True)
    def dloss(y, score):
        """Compute Squared Hinge Loss Derivative.

        Parameters
        ----------
        y : int
            True label of current sample. Must be -1/+1
        score : float
            Classification score of current sample.

        """
        z = 1 - y * score
        if z > 0:
            return -2 * y * z
        return 0.0
