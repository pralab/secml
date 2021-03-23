"""
.. module:: CRegularizerL1
   :synopsis: L1-Norm Regularizer Function

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from secml.ml.classifiers.regularizer import CRegularizer


class CRegularizerL1(CRegularizer):
    """Norm-L1 Regularizer.

    This function leads to sparse solutions.

    L1 Regularizer is given by:

    .. math::

       R(w) := \\sum_{i=1}^{n} |w_i|

    Attributes
    ----------
    class_type : 'l1'

    """
    __class_type = 'l1'

    def regularizer(self, w):
        """Returns Norm-L1.

        Parameters
        ----------
        w : CArray
            Vector-like array.

        """
        return w.norm(order=1)

    def dregularizer(self, w):
        """Returns Norm-L1 derivative.

        Parameters
        ----------
        w : CArray
            Vector-like array.

        """
        return w.sign()
