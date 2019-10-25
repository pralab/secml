"""
.. module:: CExplainerGradient
   :synopsis: Explanation of predictions via input gradient.

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml.explanation import CExplainer
from secml.array import CArray


class CExplainerGradient(CExplainer):
    """Explanation of predictions via input gradient.

    The relevance `rv` of each feature is given by:

    .. math::
       rv_i = \\frac{\\partial F(x)}{\\partial x_i}

    - D. Baehrens, T. Schroeter, S. Harmeling, M. Kawanabe, K. Hansen,
      K.-R.Muller, " "How to explain individual classification decisions",
      in: J. Mach. Learn. Res. 11 (2010) 1803-1831

    Parameters
    ----------
    clf : CClassifier
        Instance of the classifier to explain. Must be differentiable.

    Attributes
    ----------
    class_type : 'gradient'

    """
    __class_type = 'gradient'

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
        rv = grad.deepcopy()
        self.logger.debug(
            "Relevance Vector:\n{:}".format(rv))
        return (rv, grad) if return_grad is True else rv
