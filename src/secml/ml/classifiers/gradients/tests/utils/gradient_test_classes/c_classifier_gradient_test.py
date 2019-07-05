"""
.. module:: CClassifierGradientTest
   :synopsis: Debugging class for the classifier gradient class

.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
from abc import ABCMeta, abstractmethod, abstractproperty
import six

from secml.core import CCreator


@six.add_metaclass(ABCMeta)
class CClassifierGradientTest(CCreator):
    """
    This class implement different functionalities which are useful to test
    the CClassifierGradient class.
    """
    __super__ = 'CClassifierGradientTest'

    @abstractproperty
    def params(self):
        """
        Classifier parameters
        """
        raise NotImplementedError()

    @abstractmethod
    def l(self, x, y, clf):
        """
        Classifier loss
        """
        raise NotImplementedError()

    @abstractmethod
    def train_obj(self, x, y, clf):
        """
        Classifier training objective function
        """
        raise NotImplementedError()

    @abstractmethod
    def change_params(self, params, clf):
        """
        Return a deepcopy of the given classifier with the value of the
        parameters changed
        vector
        """
        raise NotImplementedError()

