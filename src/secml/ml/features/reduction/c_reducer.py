"""
.. module:: FeatureReducer
   :synopsis: Common interface for feature reduction algorithms.

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from abc import ABCMeta
import six

from secml.ml.features import CPreProcess


@six.add_metaclass(ABCMeta)
class CReducer(CPreProcess):
    """Common interface for feature reduction algorithms."""
    __super__ = 'CReducer'
