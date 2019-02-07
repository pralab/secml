"""
.. module:: CExplainer
   :synopsis: Abstract interface for Explainable ML methods.

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
from abc import ABCMeta, abstractmethod

from secml.core import CCreator


class CExplainer(CCreator):
    """Abstract interface for Explainable ML methods.

    Parameters
    ----------
    clf : CClassifier
        Instance of the classifier to explain.
    tr_ds : CDataset
        Training dataset of the classifier to explain.

    """
    __metaclass__ = ABCMeta
    __super__ = 'CExplainer'

    def __init__(self, clf, tr_ds=None):
        self._clf = clf
        self._tr_ds = tr_ds

    @property
    def clf(self):
        """Classifier to explain."""
        return self._clf

    @property
    def tr_ds(self):
        """Training dataset."""
        return self._tr_ds

    @abstractmethod
    def explain(self, *args, **kwargs):
        """Computes the explanation."""
        raise NotImplementedError
