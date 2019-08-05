"""
.. module:: CClassifierGradientRejectThresholdMixin
   :synopsis: Mixin for classifier with a reject based on a threshold gradients.

.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from secml.array import CArray
from secml.ml.classifiers.gradients import CClassifierGradientMixin


class CClassifierGradientRejectThresholdMixin(CClassifierGradientMixin):
    """Mixin class for CClassifierRejectThreshold gradients."""

    # test derivatives:

    def _grad_f_x(self, x, y):
        """Computes the gradient of the classifier's decision function
         wrt decision function input.

        The gradient taken w.r.t. the reject class can be thus set to 0,
        being its output constant regardless of the input sample x.

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
        x = x.atleast_2d()

        if y == -1:
            # the gradient is a vector with all the elements equal to zero
            return CArray.zeros(x.shape[1], sparse=x.issparse)

        elif y < self.n_classes:
            return self.clf.grad_f_x(x, y=y)

        else:
            raise ValueError("The index of the class wrt the gradient must "
                             "be computed is wrong.")
