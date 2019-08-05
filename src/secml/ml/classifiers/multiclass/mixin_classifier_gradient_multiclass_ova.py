"""
.. module:: CClassifierGradientMulticlassOVAMixin
   :synopsis: Mixin for OVA classifiers gradient.

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from secml.array import CArray
from secml.ml.classifiers.gradients import CClassifierGradientMixin


class CClassifierGradientMulticlassOVAMixin(CClassifierGradientMixin):
    """Mixin class for CClassifierMulticlassOVA gradients."""

    # test derivatives:

    def _grad_f_x(self, x, y):
        """Computes the gradient of the classifier's decision function
         wrt decision function input.

        For a multiclass OVA classifier, the gradient of the y^th
        binary classifier is returned.

        Parameters
        ----------
        x : CArray
            The gradient is computed in the neighborhood of x.
        y : int
            Index of the class wrt the gradient must be computed.
            Use -1 to output the gradient w.r.t. the reject class.

        Returns
        -------
        gradient : CArray
            Gradient of the classifier's df wrt its input. Vector-like array.

        """
        self._check_clf_index(y)  # Check the binary classifier input index
        return self._binary_classifiers[y].grad_f_x(x, y=1).ravel()
