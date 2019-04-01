from abc import ABCMeta, abstractmethod
import six

from secml.core import CCreator
from secml.array import CArray


@six.add_metaclass(ABCMeta)
class CLineSearch(CCreator):
    """Abstract class that implements line-search optimization algorithms.

    Line-search algorithms optimize the objective function along a given
    direction in the feasible domain, potentially subject to constraints.
    The search is normally stopped when the objective improves at a satisfying
    level, to keep the search fast.
    
    """
    __super__ = 'CLineSearch'

    def __init__(self, fun, constr=None, bounds=None, eta=1e-4, max_iter=20):

        # Sets the initial value of step and max number of iterations.

        self.fun = fun
        self.constr = constr
        self.bounds = bounds

        self.eta = CArray(eta)
        self.max_iter = max_iter

    @abstractmethod
    def line_search(self, fun, x, d, constr, **kwargs):
        raise NotImplementedError
