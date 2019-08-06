"""
.. module:: CSoftmax
   :synopsis: Cross Entropy Loss and Softmax function

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from secml.core import CCreator
from secml.array import CArray


class CSoftmax(CCreator):
    """Softmax function."""

    def softmax(self, s):
        """Apply the softmax function to input.

        The softmax function is defined for the vector `s`
        and for the i-th class as:

        .. math::

           \\text{SoftMax}(y, s) =
                        \\left[ a_1,\\ldots,a_n] -> [s_1,\\ldots,s_n \\right]

           where:
             \\text s_y = \\frac{e^{a_j}}{\\sum_{i=1}^N e^a_i} \\forall 1=1..N


        Parameters
        ----------
        s : CArray
            2-D array of shape (n_samples, n_classes) with input data.

        Returns
        -------
        CArray
            Softmax function. Same shape of input array.

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.ml.classifiers.loss import CSoftmax

        >>> a = CArray([[1, 2, 3], [2, 4, 5]])
        >>> print(CSoftmax().softmax(a))
        CArray([[0.090031 0.244728 0.665241]
         [0.035119 0.259496 0.705385]])

        """
        x = s.atleast_2d()  # Working with 2-D arrays only
        # this avoids numerical issues (rescaling score values to (-inf, 0])
        s_exp = (x - x.max()).exp()
        s_exp_sum = s_exp.sum(axis=1)

        return s_exp / s_exp_sum

    def gradient(self, s, y):
        """Gradient of the softmax function.

        The derivative of the y-th output of the
        softmax function w.r.t. all the inputs is given by:

        .. math::

           \\left[ \\frac{s'_y}{a'_1}, \\ldots, \\frac{s'_y}{a'_n} \\right]

           where:
             \\frac{s'_y}{a'_i} = s_y (\\delta - s_i)

           with:
             \\delta = 1 if i = j
             \\delta = 0 if i \\ne j

        Parameters
        ----------
        s : CArray
            2-D array of shape (1, n_classes) with input data.
        y : int
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

        # - sigma_{s_i} * sigma_{s_y}
        grad = -sigma_s * sigma_s[y]
        # += sigma_{s_y} if i == y
        grad[y] += sigma_s[y]

        return grad.ravel()
