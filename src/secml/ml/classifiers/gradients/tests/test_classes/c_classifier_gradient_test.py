"""
.. module:: CClassifierGradientTest
   :synopsis: Debugging class for mixin classifier gradient.

.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from abc import ABCMeta, abstractmethod

from secml.core import CCreator


class CClassifierGradientTest(CCreator, metaclass=ABCMeta):
    __super__ = 'CClassifierGradientTest'

    @abstractmethod
    def params(self, clf):
        """Classifier parameters."""
        raise NotImplementedError

    @abstractmethod
    def l(self, x, y, clf):
        """Classifier loss."""
        raise NotImplementedError

    @abstractmethod
    def train_obj(self, x, y, clf):
        """Classifier training objective function."""
        raise NotImplementedError

    @abstractmethod
    def change_params(self, params, clf):
        """Return a deepcopy of the given classifier with the value
        of the parameters changed."""
        raise NotImplementedError

