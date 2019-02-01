"""
.. module:: CClassifierGradient
   :synopsis: Common inteface for the implementations of the classifier
   gradients

.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>
"""
from abc import ABCMeta, abstractmethod

from secml.core import CCreator


class CClassifierGradient(CCreator):
    """Abstract class that defines basic methods for CClassifierGradient.
    """
    __metaclass__ = ABCMeta
    __super__ = 'CClassifierGradient'

    @abstractmethod
    def hessian(self, x, y, clf):
        """
        Compute hessian of the loss w.r.t. the classifier parameters
        """
        raise NotImplementedError()

    @abstractmethod
    def fd_params(self, x):
        """
        Derivative of the discriminant function w.r.t. the classifier
        parameters
        """
        raise NotImplementedError()

    def fd_x(self, x=None, y=1):
        """
        Derivative of the discriminant function w.r.t. an input sample
        """
        raise NotImplementedError()

    @abstractmethod
    def Ld_params(self, x):
        """
        Derivative of the classifier classifier loss function (regularizer
        included) w.r.t. the classifier parameters
        """
        raise NotImplementedError()

    @abstractmethod
    def Ld_s(self):
        """
        Derivative of the classifier classifier loss function w.r.t. the score
        """
        raise NotImplementedError
