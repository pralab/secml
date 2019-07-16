"""
.. module:: CClassifierGradientRejectDetectorMixin
   :synopsis: Mixin for classifier with a reject based on a detector gradients.

.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
from secml.ml.classifiers.gradients import CClassifierGradientMixin


class CClassifierGradientRejectDetectorMixin(CClassifierGradientMixin):
    """Mixin class for CClassifierRejectDetector gradients."""

    # test derivatives:

    def _grad_f_x(self, x, y):
        """Computes the gradient of the classifier's decision function
         wrt decision function input.

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
        if y == -1:
            # return the gradient of the detector
            # (it's binary so always return y=1)
            grad = self.det.grad_f_x(x, y=1)

            # compute the gradient of the softmax used to rescale the scores
            scores = self._det.predict(x, return_decision_function=True)[1]
            softmax_grad = self._softmax.gradient(scores, y=1)[1]

        elif y < self.n_classes:
            grad = self.clf.grad_f_x(x, y=y)

            # compute the gradient of the softmax used to rescale the scores
            scores = self.clf.predict(x, return_decision_function=True)[1]
            softmax_grad = self._softmax.gradient(scores, y=y)[y]

        else:
            raise ValueError("The index of the class wrt the gradient must "
                             "be computed is wrong.")

        return softmax_grad.item() * grad.ravel()
