"""
.. module:: CClassifierGradientTest
   :synopsis: Debugging class for the classifier gradient class

.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
from abc import ABCMeta, abstractmethod, abstractproperty

from secml.core import CCreator


class CClassifierGradientTest(CCreator):
    """
    This class implement different functionalities which are useful to test
    the CClassifierGradient class.
    """
    __metaclass__ = ABCMeta
    __super__ = 'CClassifierGradientTest'

    def __init__(self, grads_obj):
        self.grads_obj = grads_obj

    @abstractproperty
    def _params(self):
        """
        Classifier parameters
        """
        raise NotImplementedError()

    @abstractmethod
    def _L_tot(self, x, y, clf):
        """
        Classifier total loss
        L_tot = loss computed on the training samples + regularizer
        """
        raise NotImplementedError()
