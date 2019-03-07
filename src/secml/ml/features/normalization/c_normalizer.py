"""
.. module:: FeatureNormalizer
   :synopsis: Common interface for feature normalizers.

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from abc import ABCMeta

from secml.ml.features import CPreProcess


class CNormalizer(CPreProcess):
    """Common interface for normalization preprocessing algorithms."""
    __metaclass__ = ABCMeta
    __super__ = 'CNormalizer'

    def is_linear(self):
        """Returns True for linear normalizers."""
        return False
