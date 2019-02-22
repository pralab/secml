"""
.. module:: CExplainerLocal
   :synopsis: Abstract interface for Local Explainable ML methods.

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
from abc import abstractmethod

from secml.explanation import CExplainer


class CExplainerLocal(CExplainer):
    """Abstract interface for Local Explainable ML methods.

    Parameters
    ----------
    clf : CClassifier
        Instance of the classifier to explain.
    tr_ds : CDataset
        Training dataset of the classifier to explain.

    """
    @abstractmethod
    def explain(self, x, *args, **kwargs):
        """Computes the explanation for input sample."""
        raise NotImplementedError
