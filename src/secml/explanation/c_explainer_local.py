from abc import ABCMeta, abstractmethod

from secml.explanation import CExplainer


class CExplainerLocal(CExplainer):
    """Abstract interface for Local Explainable ML methods."""
    __metaclass__ = ABCMeta
    __super__ = 'CExplainerLocal'

    @abstractmethod
    def explain(self, x, *args, **kwargs):
        """Computes the explanation for input sample."""
        raise NotImplementedError
