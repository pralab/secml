"""
.. module:: Regularizer
   :synopsis: Interface for Regularizer Functions

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
from abc import ABCMeta, abstractmethod, abstractproperty
from prlib.core import CCreator


class CRegularizer(CCreator):
    """Abstract class that defines basic methods for regularizer functions."""
    __metaclass__ = ABCMeta
    __super__ = 'CRegularizer'

    @abstractproperty
    def class_type(self):
        """Defines regularizer type."""
        raise NotImplementedError()

    @abstractmethod
    def regularizer(self, *args, **kwargs):
        """Gets value of regularizer."""
        raise NotImplementedError()

    def dregularizer(self, *args, **kwargs):
        """Gets the derivative of regularizer."""
        raise NotImplementedError()
