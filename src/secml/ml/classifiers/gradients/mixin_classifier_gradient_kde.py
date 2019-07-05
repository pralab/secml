"""
.. module:: CClassifierGradientKDEMixin
   :synopsis: Mixin class for the KDE classifier gradients

.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
from abc import abstractmethod

from secml.array import CArray
from secml.ml.classifiers.gradients import CClassifierGradientMixin
from secml.ml.classifiers.clf_utils import convert_binary_labels
from secml.ml.classifiers.clf_utils import \
    check_binary_labels


class CClassifierGradientKDEMixin(CClassifierGradientMixin):
    __class_type = 'KDE'

    # required classifier properties:
    @property
    @abstractmethod
    def kernel(self):
        pass

    # train derivatives:

    def _grad_f_x(self, x=None, y=1):
        """Computes the gradient of the linear classifier's decision function
         wrt decision function input.

        For linear classifiers, the gradient wrt input is equal
        to the weights vector w. The point x can be in fact ignored.

        Parameters
        ----------
        x : CArray or None, optional
            The gradient is computed in the neighborhood of x.
        y : int, optional
            Binary index of the class wrt the gradient must be computed.
            Default is 1, corresponding to the positive class.

        Returns
        -------
        gradient : CArray
            The gradient of the linear classifier's decision function
            wrt decision function input. Vector-like array.

        """
        check_binary_labels(y)  # Label should be in {0, 1}

        k = self.kernel.gradient(self._training_samples, x)
        # Gradient sign depends on input label (0/1)
        return -convert_binary_labels(y) * k.mean(axis=0, keepdims=False)
