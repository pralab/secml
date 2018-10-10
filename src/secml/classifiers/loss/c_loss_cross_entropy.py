"""
.. module:: CLossCrossEntropy
   :synopsis: Cross Entropy Loss and SoftMax function

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
from secml.classifiers.loss import CLossClassification
from secml.array import CArray

__all__ = ['softmax', 'CLossCrossEntropy']


def softmax(x):
    """Apply the SoftMax function to input.

    The SoftMax function is defined as (for sample i):

      \text{SoftMax}(y, s) = \frac{e^{s_i}}{\sum_{k=1}^N e^s_k}

    Parameters
    ----------
    x : CArray
        2-D array with input data.

    Returns
    -------
    CArray
        SoftMax function. Same shape of input array.

    Examples
    --------
    >>> from secml.array import CArray
    >>> from secml.classifiers.loss import softmax

    >>> a = CArray([[1, 2, 3], [2, 4, 5]])
    >>> print softmax(a)
    CArray([[ 0.090031  0.244728  0.665241]
     [ 0.035119  0.259496  0.705385]])

    """
    x = x.atleast_2d()  # Working with 2-D arrays only
    # this avoids numerical issues (rescaling score values to (-inf, 0])
    s_exp = (x - x.max()).exp()
    s_exp_sum = s_exp.sum(axis=1)

    return s_exp / s_exp_sum


class CLossCrossEntropy(CLossClassification):
    """Cross Entropy Loss Function (Log Loss).

    Cross entropy indicates the distance between what the model
     believes the output distribution should be, and what the
     original distribution really is.

    The cross entropy loss is defined as (for sample i):

      L_\text{cross-entropy}(y, s) = - y_i log(\frac{e^{s_i}}{\sum_{k=1}^N e^s_k})

    Attributes
    ----------
    class_type : 'cross_entropy'
    suitable_for : 'classification'

    """
    class_type = 'cross_entropy'

    def loss(self, y_true, score, pos_label=None):
        """Computes the value of the Cross Entropy loss function.

        Parameters
        ----------
        y_true : CArray
            Ground truth (correct), targets. Vector-like array.
        score : CArray
            Outputs (predicted), targets.
            2-D array of shape (n_samples, n_classes).
        pos_label : int or None, optional
            The class wrt compute the loss function.
            Default None, meaning that the function is computed
             for each sample wrt the corresponding true label.

        Returns
        -------
        CArray
            Loss function. Vector-like array.

        """
        score = score.atleast_2d()  # Working with 2-D arrays only

        p = softmax(score)  # SoftMax function

        # find-like indexing (list of lists)
        a = y_true.tolist() if pos_label is None else [pos_label]

        return -CArray(p[[range(score.shape[0]), a]]).log()

    def dloss(self, y_true, score, pos_label=None):
        """Computes the value of the Cross Entropy loss function.

        Parameters
        ----------
        y_true : CArray
            Ground truth (correct), targets. Vector-like array.
        score : CArray
            Outputs (predicted), targets.
            2-D array of shape (n_samples, n_classes).
        pos_label : int or None, optional
            The class wrt compute the loss function.
            Default None, meaning that the function is computed
             for each sample wrt the corresponding true label.

        Returns
        -------
        CArray
            Loss function. Vector-like array.

        """
        score = score.atleast_2d()  # Working with 2-D arrays only

        grad = softmax(score)

        # find-like indexing (list of lists)
        a = y_true.tolist() if pos_label is None else [pos_label]

        grad = CArray(grad[[range(score.shape[0]), a]])

        return grad - 1.0
