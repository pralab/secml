"""
.. module:: SquaredLoss
   :synopsis: Squared Loss Function

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
from prlib.classifiers.loss import CLoss


class CLossSquared(CLoss):
    """Squared Loss Function.

    Ordinary least squares fit.

    """

    class_type = 'squared_loss'
    loss_type = 'regression'

    def loss(self, y, score):
        """Compute Squared Loss.

        Parameters
        ----------
        y : CArray
            Vector-like array.
        score : CArray
            Vector-like array.

        """
        return 0.5 * ((score - y) ** 2)

    def dloss(self, y, score):
        """Compute Squared Loss Derivative.

        Parameters
        ----------
        y : CArray
            Vector-like array.
        score : CArray
            Vector-like array.

        """
        return score - y
