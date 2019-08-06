"""
.. module:: CNormalizer
   :synopsis: Interface for feature normalizers.

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from abc import ABCMeta
import six

from secml.ml.features import CPreProcess


@six.add_metaclass(ABCMeta)
class CNormalizer(CPreProcess):
    """Common interface for normalization preprocessing algorithms."""
    __super__ = 'CNormalizer'

    def is_linear(self):
        """Returns True for linear normalizers."""
        return False
