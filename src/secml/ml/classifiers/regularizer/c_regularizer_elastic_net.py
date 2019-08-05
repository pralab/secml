"""
.. module:: CRegularizerElasticNet
   :synopsis: ElasticNet Regularizer Function

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from secml.ml.classifiers.regularizer import CRegularizer


class CRegularizerElasticNet(CRegularizer):
    """ElasticNet Regularizer.

    A convex combination of L2 and L1,
    where :math:`\\rho` is given by `1 - l1_ratio`.

    ElasticNet Regularizer is given by:

    .. math::

        R(w) := \\frac{\\rho}{2} \\sum_{i=1}^{n} w_i^2 + (1-\\rho)
                                 \\sum_{i=1}^{n} |w_i|

    Attributes
    ----------
    class_type : 'elastic-net'

    """
    __class_type = 'elastic-net'

    def __init__(self, l1_ratio=0.15):
        self._l1_ratio = float(l1_ratio)

    @property
    def l1_ratio(self):
        """Get l1-ratio."""
        return self._l1_ratio

    @l1_ratio.setter
    def l1_ratio(self, value):
        """Set l1-ratio (float)."""
        self._l1_ratio = float(value)

    def regularizer(self, w):
        """Returns ElasticNet Regularizer.

        Parameters
        ----------
        w : CArray
            Vector-like array.

        """
        return self.l1_ratio * w.norm(order=1) \
            + (1 - self.l1_ratio) * 0.5 * (w ** 2).sum()

    def dregularizer(self, w):
        """Returns the derivative of the elastic-net regularizer

        Parameters
        ----------
        w : CArray
            Vector-like array.

        """
        return self.l1_ratio * w.sign() + (1 - self.l1_ratio) * w
