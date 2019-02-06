from abc import ABCMeta, abstractmethod, abstractproperty
from secml.core import CCreator

class CExplainer(CCreator):
    """
    Abstract class for the explainability methods.
    """
    __metaclass__ = ABCMeta
    __super__ = 'CExplainer'

    def __init__(self, clf):
        self._clf = clf

    def explain(self):
        NotImplementedError()