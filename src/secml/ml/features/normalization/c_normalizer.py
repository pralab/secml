"""
.. module:: CNormalizer
   :synopsis: Interface for feature normalizers.

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from abc import ABCMeta

from secml.core.decorators import deprecated
from secml.ml.features import CPreProcess


class CNormalizer(CPreProcess, metaclass=ABCMeta):
    """Common interface for normalization preprocessing algorithms."""
    __super__ = 'CNormalizer'

    @deprecated('0.11')
    def is_linear(self):
        """Returns True for linear normalizers."""
        return False
