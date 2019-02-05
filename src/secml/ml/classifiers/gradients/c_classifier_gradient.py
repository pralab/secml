"""
.. module:: CClassifierGradient
   :synopsis: Common interface for the implementations of the
              classifier gradients

.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
from abc import ABCMeta, abstractmethod

from secml.core import CCreator


class CClassifierGradient(CCreator):
    """Abstract class that defines basic methods for CClassifierGradient.
    """
    __metaclass__ = ABCMeta
    __super__ = 'CClassifierGradient'

    def loss(self):
        return self._loss

    @abstractmethod
    def hessian(self, clf, x, y):
        """
        Compute hessian of the loss w.r.t. the classifier parameters
        """
        raise NotImplementedError

    @abstractmethod
    def fd_params(self, clf, x):
        """
        Derivative of the discriminant function w.r.t. the classifier
        parameters
        """
        raise NotImplementedError

    def fd_x(self, clf, x=None, y=1):
        """
        Derivative of the discriminant function w.r.t. an input sample
        """
        raise NotImplementedError

    @abstractmethod
    def L_d_params(self, clf, x, y, loss=None, regularized=True):
        """
        Derivative of the classifier loss function

        Parameters
        ----------
        x : CArray
            features of the dataset on which the loss is computed
        y :  CArray
            features of the training samples
        loss: None (default) or CLoss
            If the loss is equal to None (default) the classifier loss is used
            to compute the derivative.
        regularized: boolean
            If True (default) the loss on which the derivative is computed
            is the loss on the given samples + the regularizer,
            which is not considered if the argument is False
        """
        raise NotImplementedError

    @abstractmethod
    def L_tot(self, x, y, clf):
        """
        Classifier total loss
        L_tot = loss computed on the training samples + regularizer
        """
        raise NotImplementedError()

    def Ld_s(self, w, y, score):
        """
        Derivative of the classifier loss function w.r.t. the score
        """
        return self.loss.dloss(y, score)
