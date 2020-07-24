"""
.. module:: CClassifierLinear
   :synopsis: Interface and common functions for linear classification

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>
.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>

"""
from abc import abstractmethod

from secml.array import CArray


class CClassifierLinearMixin:
    """Mixin class that defines basic methods for linear classifiers.

    A linear classifier assigns a label (class) to new patterns
    computing the inner product between the patterns and a vector
    of weights for each training set feature.

    This interface defines the weight and bias, and the forward and backward
    functions for linear classifiers.

    """

    @property
    @abstractmethod
    def w(self):
        """Vector with feature weights (dense or sparse)."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def b(self):
        """Bias term."""
        raise NotImplementedError()

    def _forward(self, x):
        """Compute scores proportionally to the distance to the
        hyperplane as w'x + b.

        Parameters
        ----------
        x : CArray
            Input samples given as matrix with shape=(n_samples, n_features).

        Returns
        -------
        score : CArray
            Value of the decision function for each sample, given as a matrix
            with shape=(n_samples, n_classes).

        """
        score = CArray(x.dot(self.w.T)).todense().ravel() + self.b
        scores = CArray.ones(shape=(x.shape[0], 2))
        scores[:, 0] = -score.ravel().T
        scores[:, 1] = score.ravel().T
        return scores

    def _backward(self, w):
        """Computes the gradient of the linear classifier's decision function
         wrt decision function input.

        For linear classifiers, the gradient wrt input is equal
        to the weights vector w. The point x can be in fact ignored.

        Parameters
        ----------
        w : CArray
            The vector to be pre-multiplied (reverse mode diff)

        Returns
        -------
        gradient : CArray
            The gradient of the linear classifier's decision function
            wrt decision function input. Vector-like array.

        """
        # Gradient sign depends on input label (0/1)
        if w is not None:
            return w[0] * -self.w + w[1] * self.w
        else:
            raise ValueError("w cannot be set as None.")
