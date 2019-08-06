"""
.. module:: CClassifierGradientKDEMixin
   :synopsis: Mixin for KDE classifier gradients.

.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from secml.array import CArray
from secml.ml.classifiers.gradients import CClassifierGradientMixin
from secml.ml.classifiers.clf_utils import convert_binary_labels


class CClassifierGradientKDEMixin(CClassifierGradientMixin):
    """Mixin class for CClassifierKDE gradients."""

    # train derivatives:

    def _grad_f_x(self, x, y=1):
        """Computes the gradient of the classifier's decision function
         wrt decision function input.

        Parameters
        ----------
        x : CArray
            The gradient is computed in the neighborhood of x.
        y : int, optional
            Binary index of the class wrt the gradient must be computed.
            Default is 1, corresponding to the positive class.

        Returns
        -------
        gradient : CArray
            The gradient of the classifier's decision function
            wrt decision function input. Vector-like array.

        """
        k = self.kernel.gradient(self._training_samples, x)
        grad = k.mean(axis=0, keepdims=False)
        grad = grad.tosparse() if k.issparse else grad
        # Gradient sign depends on input label (0/1)
        return -convert_binary_labels(y) * grad
