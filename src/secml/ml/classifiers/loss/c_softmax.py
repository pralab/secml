"""
.. module:: CSoftmax
   :synopsis: Cross Entropy Loss and SoftMax function

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from secml.core import CCreator
from secml.array import CArray


class CSoftmax(CCreator):
    """SoftMax function."""

    def softmax(self, x):
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
        >>> from secml.ml.classifiers.loss import CSoftmax

        >>> a = CArray([[1, 2, 3], [2, 4, 5]])
        >>> print CSoftmax().softmax(a)
        CArray([[ 0.090031  0.244728  0.665241]
         [ 0.035119  0.259496  0.705385]])

        """
        x = x.atleast_2d()  # Working with 2-D arrays only
        # this avoids numerical issues (rescaling score values to (-inf, 0])
        s_exp = (x - x.max()).exp()
        s_exp_sum = s_exp.sum(axis=1)

        return s_exp / s_exp_sum

    def gradient(self, x, pos_label=None):
        """Gradient of the softmax function."""
        raise NotImplementedError
