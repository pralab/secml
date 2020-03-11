"""
.. module:: CClassifierGradientSGDMixin
   :synopsis: Mixin for SGD classifier gradients.

.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml.array import CArray
from secml.ml.classifiers.gradients import CClassifierGradientLinearMixin


class CClassifierGradientSGDMixin(CClassifierGradientLinearMixin):
    """Mixin class for CClassifierSGD gradients."""

    # train derivatives:

    def grad_tr_params(self, x, y):
        """
        Derivative of the classifier training objective function w.r.t. the
        classifier parameters

        Parameters
        ----------
        x : CArray
            Features of the dataset on which the training objective is computed.
        y : CArray
            dataset labels

        """
        raise NotImplementedError
