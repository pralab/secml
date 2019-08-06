"""
.. module:: CLossEpsilonInsensitive
   :synopsis: Epsilon-Insensitive Loss Functions

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from secml.ml.classifiers.loss import CLossRegression
from secml.array import CArray


class CLossEpsilonInsensitive(CLossRegression):
    """Epsilon-Insensitive Loss Function.

    Any difference between the current prediction and
    the ground truth is ignored if is less than the
    `epsilon` threshold.

    Epsilon-Insensitive loss is used by support vector regression.

    The Epsilon-Insensitive loss is defined as:

    .. math::

       L_{\\epsilon-\\text{ins}} (y, s) =
                             \\max \\left\\{ |y - s| - \\epsilon, 0 \\right\\}

    Attributes
    ----------
    class_type : 'e-insensitive'
    suitable_for : 'regression'

    """
    __class_type = 'e-insensitive'

    def __init__(self, epsilon=0.1):
        self._epsilon = float(epsilon)

    @property
    def epsilon(self):
        """Threshold parameter epsilon."""
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        """Set the threshold parameter epsilon."""
        self._epsilon = float(value)

    def loss(self, y_true, score):
        """Computes the value of the epsilon-insensitive loss function.

        Parameters
        ----------
        y_true : CArray
            Ground truth (correct), targets. Vector-like array.
        score : CArray
            Outputs (predicted), targets.
            Vector-like array of shape (n_samples,).

        Returns
        -------
        CArray
            Loss function. Vector-like array.

        """
        if score.is_vector_like is False:
            raise ValueError("only a vector-like `score` array is supported.")

        # Ensure we work with vector-like arrays
        y_true = y_true.ravel()
        score = score.ravel()

        # max(0, abs(y - s) - epsilon)
        e = abs(y_true - score) - self.epsilon
        e[e < 0] = 0.0

        return e

    def dloss(self, y_true, score):
        """Computes the derivative of the epsilon-insensitive loss function
         with respect to `score`.

        Parameters
        ----------
        y_true : CArray
            Ground truth (correct), targets. Vector-like array.
        score : CArray
            Outputs (predicted), targets.
            Vector-like array of shape (n_samples,).

        Returns
        -------
        CArray
            Derivative of the loss function. Vector-like array.

        """
        if score.is_vector_like is False:
            raise ValueError("only a vector-like `score` array is supported.")

        # Ensure we work with vector-like arrays
        y_true = y_true.ravel()
        score = score.ravel()

        # -1 if (y - s) > epsilon, 1 if (y - s) < -epsilon, 0 otherwise
        d = CArray.zeros(shape=y_true.size, dtype=float)
        d[y_true - score > self.epsilon] = -1
        d[score - y_true > self.epsilon] = 1

        return d


class CLossEpsilonInsensitiveSquared(CLossEpsilonInsensitive):
    """Squared Epsilon-Insensitive Loss Function.

    Any difference between the current prediction and
    the ground truth is ignored if is less than the
    `epsilon` threshold.

    The Squared Epsilon-Insensitive loss is defined as:

    .. math::

       L^2_{\\epsilon-\\text{ins}} (y, s) =
        {\\left( \\max\\left\\{ |y - s| - \\epsilon, 0 \\right\\} \\right)}^2

    Attributes
    ----------
    class_type : 'e-insensitive-squared'
    suitable_for : 'regression'

    """
    __class_type = 'e-insensitive-squared'

    def loss(self, y_true, score):
        """Computes the value of the squared epsilon-insensitive loss function.

        Parameters
        ----------
        y_true : CArray
            Ground truth (correct), targets. Vector-like array.
        score : CArray
            Outputs (predicted), targets.
            Vector-like array of shape (n_samples,).

        Returns
        -------
        CArray
            Loss function. Vector-like array.

        """
        if score.is_vector_like is False:
            raise ValueError("only a vector-like `score` array is supported.")

        # Ensure we work with vector-like arrays
        y_true = y_true.ravel()
        score = score.ravel()

        # (max(0, abs(y - s) - epsilon))^2
        e = abs(y_true - score) - self.epsilon
        e2 = e ** 2
        e2[e < 0] = 0

        return e2

    def dloss(self, y_true, score):
        """Computes the derivative of the squared epsilon-insensitive
         loss function with respect to `score`.

        Parameters
        ----------
        y_true : CArray
            Ground truth (correct), targets. Vector-like array.
        score : CArray
            Outputs (predicted), targets.
            Vector-like array of shape (n_samples,).

        Returns
        -------
        CArray
            Derivative of the loss function. Vector-like array.

        """
        if score.is_vector_like is False:
            raise ValueError("only a vector-like `score` array is supported.")

        # Ensure we work with vector-like arrays
        y_true = y_true.ravel()
        score = score.ravel()

        d = CArray.zeros(shape=y_true.size, dtype=float)
        z = y_true - score
        d[z > self.epsilon] = -2 * (z[z > self.epsilon] - self.epsilon)
        d[z < self.epsilon] = 2 * (-z[z < self.epsilon] - self.epsilon)
        return d
