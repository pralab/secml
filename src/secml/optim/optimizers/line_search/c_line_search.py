"""
.. module:: CLineSearch
   :synopsis: Interface for line search methods.

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>

"""
from abc import ABCMeta, abstractmethod

from secml.core import CCreator
from secml.array import CArray


class CLineSearch(CCreator, metaclass=ABCMeta):
    """Abstract class that implements line-search optimization algorithms.

    Line-search algorithms optimize the objective function along a given
    direction in the feasible domain, potentially subject to constraints.
    The search is normally stopped when the objective improves at a satisfying
    level, to keep the search fast.

    Parameters
    ----------
    fun : CFunction
        The function to use for the optimization.
    constr : CConstraintL1 or CConstraintL2 or None, optional
        A distance constraint. Default None.
    bounds : CConstraintBox or None, optional
        A box constraint. Default None.
    eta : scalar, optional
        Minimum resolution of the line-search grid. Default 1e-4.
    max_iter : int, optional
        Maximum number of iterations of the line search. Default 20.
    
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
    def minimize(self, x, d, **kwargs):
        """Line search.

        Parameters
        ----------
        x : CArray
            The input point.
        d : CArray
            The descent direction along which fun(x) is minimized.
        kwargs : dict
            Additional parameters required to evaluate `fun(x, **kwargs)`.

        """
        raise NotImplementedError
