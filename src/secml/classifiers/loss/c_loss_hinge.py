"""
.. module:: HingeLoss
   :synopsis: Hinge Loss Function

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from secml.classifiers.loss import CLoss
from secml.classifiers.clf_utils import extend_binary_labels
from secml.array import CArray


class CLossHinge(CLoss):
    """Hinge Loss Function (soft-margin).

    Useful to construct Support Vector Machines.

    """

    class_type = 'hinge'
    loss_type = 'classification'

    def loss(self, y, score):
        """Compute Hinge Loss.

        Parameters
        ----------
        y : CArray
            Vector-like array.
        score : CArray
            Vector-like array.

        """
        y = extend_binary_labels(y)
        h = CArray(1.0 - y * score)
        h[h < 0] = 0.0
        return h

    def dloss(self, y, score):
        """Compute Hinge Loss Derivative.

        Parameters
        ----------
        y : CArray
            Vector-like array.
        score : CArray
            Vector-like array.

        """
        y = extend_binary_labels(y)
        h = 1.0 - y * score
        d = CArray(-y.astype(float))
        d[h < 0] = 0.0
        return d


class CLossSquaredHinge(CLoss):
    """Squared Hinge Loss Function (soft-margin).

    Useful to construct Support Vector Machines.

    """

    class_type = 'squared_hinge'
    loss_type = 'classification'

    def loss(self, y, score):
        """Compute Squared Hinge Loss.

        Parameters
        ----------
        y : CArray
            Vector-like array.
        score : CArray
            Vector-like array.

        """
        y = extend_binary_labels(y)
        h = 1.0 - y * score
        h[h < 0] = 0.0
        return h ** 2

    def dloss(self, y, score):
        """Compute Squared Hinge Loss Derivative.

        Parameters
        ----------
        y : CArray
            Vector-like array.
        score : CArray
            Vector-like array.

        """
        y = extend_binary_labels(y)
        d = -2 * (y - score)
        d[1 - y * score < 0] = 0.0
        return d.astype(float)
