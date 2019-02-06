from abc import ABCMeta

from secml.explanation import CExplainer


class CExplainerLocal(CExplainer):
    """
    Abstract class for the explainability methods that provide local
    explanations.
    """
    __metaclass__ = ABCMeta
    __super__ = 'CExplainerLocal'
