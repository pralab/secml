"""
.. module:: Loss
   :synopsis: Interface for Loss Functions

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
from abc import ABCMeta, abstractmethod, abstractproperty
from secml.core import CCreator


class CLoss(CCreator):
    """Abstract class that defines basic methods for loss functions.

    Attributes
    ----------
    usenumba : True if class uses Numba for optimization.

    """
    __metaclass__ = ABCMeta
    __super__ = 'CLoss'

    usenumba = False

    @abstractproperty
    def class_type(self):
        """Defines the name of the inherited loss function class
           to dynamically instantiate it.
           Example: CLoss.create('hinge')."""
        raise NotImplementedError()

    @abstractproperty
    def loss_type(self):
        """Defines whether the loss is suitable for
           classification or regression problems."""
        raise NotImplementedError()

    @abstractmethod
    def loss(self, y, score):
        """Computes the value of the loss function loss(y,f(x))."""
        raise NotImplementedError()

    def dloss(self, y, score):
        """Computes the derivative of the loss function l(y,f(x))
           with respect to the value of f(x)."""
        raise NotImplementedError()
