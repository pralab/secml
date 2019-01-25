"""
.. module:: CSoftmax
   :synopsis: Cross Entropy Loss and SoftMax function

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
from secml.core import CCreator
from secml.array import CArray


class CSoftmax(CCreator):
    """SoftMax function."""

    def softmax(self, s):
        """Apply the SoftMax function to input.

        The SoftMax function is defined for a single sample
        and for the i-th class as:

          \text{SoftMax}(y, s) = \frac{e^{s_i}}{\sum_{k=1}^N e^s_k}

        Parameters
        ----------
        s : CArray
            2-D array of shape (n_samples, n_classes) with input data.

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
        x = s.atleast_2d()  # Working with 2-D arrays only
        # this avoids numerical issues (rescaling score values to (-inf, 0])
        s_exp = (x - x.max()).exp()
        s_exp_sum = s_exp.sum(axis=1)

        return s_exp / s_exp_sum

    def gradient(self, s, pos_label):
        """Gradient of the softmax function.

        The derivative of the i-th element of the vector s is:

            sigma = softmax(s);
            grad_i = sigma_{s_\text{pos_label}} * (t - sigma_{s_i})
                            where t = 1 if i == pos_label, t = 0 elsewhere

        Parameters
        ----------
        s : CArray
            2-D array of shape (1, n_classes) with input data.
        pos_label : int
            The class wrt compute the gradient.

        Returns
        -------
        CArray
            Softmax function gradient. Vector-like array.

        """
        if not s.is_vector_like:
            raise ValueError(
                "gradient can be computed for a single point only")

        sigma_s = self.softmax(s)

        # - sigma_{s_i} * sigma_{s_\text{pos_label}}
        grad = -sigma_s * sigma_s[pos_label]
        # += sigma_{s_\text{pos_label}} if i == pos_label
        grad[pos_label] += sigma_s[pos_label]

        return grad.ravel()
