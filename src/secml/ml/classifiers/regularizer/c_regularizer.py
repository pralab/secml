"""
.. module:: CRegularizer
   :synopsis: Interface for Regularizer Functions

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from abc import ABCMeta, abstractmethod
import six

from secml.core import CCreator


@six.add_metaclass(ABCMeta)
class CRegularizer(CCreator):
    """Abstract class that defines basic methods for regularizer functions."""
    __super__ = 'CRegularizer'

    @abstractmethod
    def regularizer(self, *args, **kwargs):
        """Gets value of regularizer."""
        raise NotImplementedError()

    def dregularizer(self, *args, **kwargs):
        """Gets the derivative of regularizer."""
        raise NotImplementedError()
