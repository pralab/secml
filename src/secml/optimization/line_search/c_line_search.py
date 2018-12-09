from abc import ABCMeta, abstractmethod, abstractproperty

from secml.core import CCreator
from secml.array import CArray


class CLineSearch(CCreator):
    """
    Abstract class to implement line searches.
    Subclasses explore some points on a segment to find out 
    that who minimizes the function of the object fun.
    It is possibile that the search is subject to constraints
    """
    metaclass__ = ABCMeta
    __super__ = 'CLineSearch'

    def __init__(self, fun, constr=None, bounds=None, eta=1e-4, max_iter=20):
        """
        Sets the initial value of step and max number of iterations
        """

        self.fun = fun
        self.constr = constr
        self.bounds = bounds

        self.eta = CArray(eta)
        self.max_iter = max_iter

    @abstractmethod
    def line_search(self, fun, x, d, constr, **kwargs):
        raise NotImplementedError()
