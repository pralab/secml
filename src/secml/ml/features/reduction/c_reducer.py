"""
.. module:: CReducer
   :synopsis: Interface for feature dimensionality reduction algorithms.

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from abc import ABCMeta
import six

from secml.ml.features import CPreProcess


@six.add_metaclass(ABCMeta)
class CReducer(CPreProcess):
    """Interface for feature dimensionality reduction algorithms."""
    __super__ = 'CReducer'
