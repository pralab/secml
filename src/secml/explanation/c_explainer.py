"""
.. module:: CExplainer
   :synopsis: Abstract interface for Explainable ML methods.

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from abc import ABCMeta, abstractmethod

from secml.core import CCreator


class CExplainer(CCreator, metaclass=ABCMeta):
    """Abstract interface for Explainable ML methods.

    Parameters
    ----------
    clf : CClassifier
        Instance of the classifier to explain.

    """
    __super__ = 'CExplainer'

    def __init__(self, clf):
        self._clf = clf

    @property
    def clf(self):
        """Classifier to explain."""
        return self._clf

    @abstractmethod
    def explain(self, x, *args, **kwargs):
        """Computes the explanation on x."""
        raise NotImplementedError
