"""
.. module:: CReducer
   :synopsis: Interface for feature dimensionality reduction algorithms.

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from abc import ABCMeta

from secml.ml.features import CPreProcess


class CReducer(CPreProcess, metaclass=ABCMeta):
    """Interface for feature dimensionality reduction algorithms."""
    __super__ = 'CReducer'
