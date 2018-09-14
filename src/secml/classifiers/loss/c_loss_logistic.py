"""
.. module:: Logistic loss
   :synopsis: Logistic loss (log loss, cross-entropy loss)

.. moduleauthor:: Battista Biggio <battista.biggio@diee.unica.it>

"""
from prlib.classifiers.loss import CLoss
from prlib.classifiers.clf_utils import extend_binary_labels
from prlib.array import CArray


class CLossLogistic(CLoss):
    """Hinge Loss Function (soft-margin).

    Useful to construct Support Vector Machines.

    """

    class_type = 'logistic'
    loss_type = 'classification'

    def __init__(self, extend_binary_labels=True):
        # this maps standard binary labels {0, 1} to {-1, 1}
        self._extend_binary_labels = bool(extend_binary_labels)

        # set upper bound for a linear approximation when -yf is large
        # to avoid numerical overflows
        # the linear approximation is fine beyond 10:
        #   log(1+exp(10)) = 10.000045
        self._bound = 10


    def loss(self, y, score):
        """Compute Logistic Loss.
        log ( 1+ exp(-yf))

        Parameters
        ----------
        y : CArray
            Vector-like array.
        score : CArray
            Vector-like array.

        """
        if self._extend_binary_labels:
            y = extend_binary_labels(y)

        v = CArray(- y * score).astype(float)

        if self._bound is None:
            h = (1.0 + v.exp()).log()
        else:
            # linear approximation avoids numerical overflows
            h = v.astype(float)  # when -yf >> 1 : log ( 1+ exp(-yf)) ~= -yf
            h[v < self._bound] = (1.0 + CArray(v[v < self._bound]).exp()).log()
        return h

    def dloss(self, y, score):
        """Compute Loss Derivative w.r.t. f
        d log ( 1+ exp(-yf)) / df =
            1/( 1+ exp(-yf)) exp(-yf) -y

        Parameters
        ----------
        y : CArray
            Vector-like array.
        score : CArray
            Vector-like array.

        """

        if self._extend_binary_labels:
            y = extend_binary_labels(y)

        v = CArray(- y * score).astype(float)

        if self._bound is None:
            h = -y * v.exp() / (1.0 + v.exp())
        else:
            if v.size > 1:
                # linear approximation avoids numerical overflows
                # when -yf >> 1 : loss ~= -yf, and grad = -y
                h = -y.astype(float)
                h[v < self._bound] = -y[v < self._bound] * \
                                     CArray(v[v < self._bound]).exp() / \
                                     (1.0 + CArray(v[v < self._bound]).exp())
            else:
                # single point
                h = -y * v.exp() / (1.0 + v.exp()) if v < self._bound else -y
        return h
