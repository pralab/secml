"""
.. module:: CLossLogistic
   :synopsis: Logistic loss function

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml.ml.classifiers.loss import CLossClassification
from secml.ml.classifiers.loss.c_loss import _check_binary_score
from secml.ml.classifiers.clf_utils import convert_binary_labels
from secml.array import CArray


class CLossLogistic(CLossClassification):
    """Logistic loss function.

    Attributes
    ----------
    class_type : 'log'
    suitable_for : 'classification'

    """
    __class_type = 'log'

    def loss(self, y_true, score, pos_label=1, bound=10):
        """Computes the value of the logistic loss function.

        Parameters
        ----------
        y_true : CArray
            Ground truth (correct), targets. Vector-like array.
        score : CArray
            Outputs (predicted), targets.
            2-D array of shape (n_samples, n_classes) or 1-D flat array
            of shape (n_samples,). If 1-D array, the probabilities
            provided are assumed to be that of the positive class.
        pos_label : {0, 1}, optional
            The class wrt compute the loss function. Default 1.
            If `score` is a 1-D flat array, this parameter is ignored.
        bound : scalar or None, optional
            Set an upper bound for a linear approximation when -y*s is large
            to avoid numerical overflows.
            10 is a generally acceptable -> log(1+exp(10)) = 10.000045

        Returns
        -------
        CArray
            Loss function. Vector-like array.

        """
        if pos_label not in (0, 1):
            raise ValueError("only {0, 1} are accepted for `pos_label`")

        y_true = convert_binary_labels(y_true).ravel()  # Convert to {-1, 1}
        score = _check_binary_score(score, pos_label)

        # log(1 + exp(-y*s)) / log(2)
        v = CArray(- y_true * score).astype(float)

        if bound is None:
            v = (1.0 + v.exp()).log()

        else:
            # linear approximation avoids numerical overflows
            # when -yf >> 1 : log ( 1+ exp(-yf)) ~= -yf
            v[v < bound] = (1.0 + v[v < bound].exp()).log()

        return v / CArray([2]).log()

    def dloss(self, y_true, score, pos_label=1, bound=10):
        """Computes the derivative of the hinge loss function with respect to `score`.

        Parameters
        ----------
        y_true : CArray
            Ground truth (correct), targets. Vector-like array.
        score : CArray
            Outputs (predicted), targets.
            2-D array of shape (n_samples, n_classes) or 1-D flat array
            of shape (n_samples,). If 1-D array, the probabilities
            provided are assumed to be that of the positive class.
        pos_label : {0, 1}, optional
            The class wrt compute the loss function derivative. Default 1.
            If `score` is a 1-D flat array, this parameter is ignored.
        bound : scalar or None, optional
            Set an upper bound for a linear approximation when -y*s is large
            to avoid numerical overflows.
            10 is a generally acceptable -> log(1+exp(10)) = 10.000045

        Returns
        -------
        CArray
            Derivative of the loss function. Vector-like array.

        """
        if pos_label not in (0, 1):
            raise ValueError("only {0, 1} are accepted for `pos_label`")

        y_true = convert_binary_labels(y_true).ravel()  # Convert to {-1, 1}
        score = _check_binary_score(score, pos_label)

        # d/df log ( 1+ exp(-yf)) / log(2)  =
        #     1/ log(2) * ( 1+ exp(-yf)) exp(-yf) -y

        v = CArray(- y_true * score).astype(float)

        if bound is None:
            h = -y_true * v.exp() / (1.0 + v.exp())

        else:
            # linear approximation avoids numerical overflows
            # when -yf >> 1 : loss ~= -yf, and grad = -y
            h = -y_true.astype(float)
            h[v < bound] = h[v < bound] * v[v < bound].exp() / \
                                                    (1.0 + v[v < bound].exp())

        return h / CArray([2]).log()
