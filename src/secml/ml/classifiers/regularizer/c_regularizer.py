"""
.. module:: CRegularizer
   :synopsis: Interface for Regularizer Functions

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from abc import ABCMeta, abstractmethod

from secml.core import CCreator


class CRegularizer(CCreator, metaclass=ABCMeta):
    """Abstract class that defines basic methods for regularizer functions."""
    __super__ = 'CRegularizer'

    @abstractmethod
    def regularizer(self, *args, **kwargs):
        """Gets value of regularizer."""
        raise NotImplementedError()

    def dregularizer(self, *args, **kwargs):
        """Gets the derivative of regularizer."""
        raise NotImplementedError()
