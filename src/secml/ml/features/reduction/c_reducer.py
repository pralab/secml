"""
.. module:: FeatureReducer
   :synopsis: Common interface for feature reduction algorithms.

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from abc import ABCMeta

from secml.ml.features import CPreProcess


class CReducer(CPreProcess):
    """Common interface for feature reduction algorithms."""
    __metaclass__ = ABCMeta
    __super__ = 'CReducer'
