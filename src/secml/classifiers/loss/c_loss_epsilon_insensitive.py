"""
.. module:: Epsilon-Insensitive
   :synopsis: Epsilon-Insensitive Loss Function

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
from prlib.classifiers.loss import CLoss
from prlib.array import CArray


class CLossEpsilonInsensitive(CLoss):
    """Epsilon-Insensitive Loss Function (soft-margin).

    Useful to construct Support Vector Regression.

    Any differences between the current prediction and
    the correct label are ignored if they are less than
    `epsilon` threshold.

    """

    class_type = 'epsilon_insensitive'
    loss_type = 'regression'

    # TODO: extend binary labels
    def __init__(self, epsilon=0.1):
        self._epsilon = float(epsilon)

    @property
    def epsilon(self):
        """Get Epsilon Parameter"""
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        """Set Epsilon Value"""
        self._epsilon = float(value)

    def loss(self, y, score):
        """Compute Epsilon Insensitive Loss.

        `loss = max(0, |y - p| - epsilon)`

        Parameters
        ----------
        y : CArray
            Vector-like array.
        score : CArray
            Vector-like array.

        """
        l = abs(y - score) - self.epsilon
        l[l < 0] = 0
        return l

    def dloss(self, y, score):
        """Compute Epsilon Insensitive Loss Derivative.

        Parameters
        ----------
        y : CArray
            Vector-like array.
        score : CArray
            Vector-like array.

        """
        d = CArray.zeros(shape=y.size, dtype=int)
        d[y - score > self.epsilon] = -1
        d[score - y > self.epsilon] = 1
        return d


class CLossSquaredEpsilonInsensitive(CLoss):
    """Squared Epsilon-Insensitive Loss Function (soft-margin).

    Useful to construct Support Vector Regression.

    Any differences between the current prediction and
    the correct label are ignored if they are less than
    `epsilon` threshold.

    """

    class_type = 'squared_epsilon_insensitive'
    loss_type = 'regression'

    def __init__(self, epsilon=0.1):
        self._epsilon = float(epsilon)

    @property
    def epsilon(self):
        """Get Epsilon Parameter"""
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        """Set Epsilon Value"""
        self._epsilon = float(value)

    def loss(self, y, score):
        """Compute Squared Epsilon Insensitive Loss.

        `loss = max(0, |y - p| - epsilon)^2`

        Parameters
        ----------
        y : CArray
            Vector-like array.
        score : CArray
            Vector-like array.

        """
        l = abs(y - score) - self.epsilon
        l2 = l ** 2
        l2[l < 0] = 0
        return l2

    def dloss(self, y, score):
        """Compute Squared Epsilon Insensitive Loss Derivative.

        Parameters
        ----------
        y : CArray
            Vector-like array.
        score : CArray
            Vector-like array.

        """
        d = CArray.zeros(shape=y.size, dtype=int)
        z = y - score
        d[z > self.epsilon] = -2 * (z[z > self.epsilon] - self.epsilon)
        d[z < self.epsilon] = 2 * (-z[z < self.epsilon] - self.epsilon)
        return d
