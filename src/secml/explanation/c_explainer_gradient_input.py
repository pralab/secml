"""
.. module:: CExplainerGradientInput
   :synopsis: Explanation of predictions via gradient*input vector.

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml.array import CArray

from secml.explanation import CExplainerGradient


class CExplainerGradientInput(CExplainerGradient):
    """Explanation of predictions via gradient*input vector.

    The relevance `rv` of each features is given by:

    .. math::
       rv_i(x) = \\left(x_i * \\frac{\\partial F(x)}{\\partial x_i}\\right)

    - A. Shrikumar, P. Greenside, A. Shcherbina, A. Kundaje,
      "Not just a blackbox: Learning important features through propagating
      activation differences", 2016 arXiv:1605.01713.
    - M. Melis, D. Maiorca, B. Biggio, G. Giacinto and F. Roli,
      "Explaining Black-box Android Malware Detection,"
      2018 26th European Signal Processing Conference (EUSIPCO),
      Rome, 2018, pp. 524-528.

    Parameters
    ----------
    clf : CClassifier
        Instance of the classifier to explain. Must be differentiable.

    Attributes
    ----------
    class_type : 'gradient-input'

    """
    __class_type = 'gradient-input'

    def explain(self, x, y, return_grad=False):
        """Computes the explanation for input sample.

        Parameters
        ----------
        x : CArray
            Input sample.
        y : int
            Class wrt compute the classifier gradient.
        return_grad : bool, optional
            If True, also return the clf gradient computed on x. Default False.

        Returns
        -------
        relevance : CArray
            Relevance vector for input sample.

        """
        grad = self.clf.grad_f_x(x, y=y)
        rv = x * grad  # Directional derivative
        self.logger.debug(
            "Relevance Vector:\n{:}".format(rv))
        return (rv, grad) if return_grad is True else rv
