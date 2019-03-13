"""
.. module:: COptimizer
   :synopsis: Interface for function optimization and minimization.

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from secml.core import CCreator
from secml.optimization.function import CFunction

from abc import ABCMeta, abstractmethod


class COptimizer(CCreator):
    """
    Class serving as interface to define optimization problems in the form:

    minimize f(x)
    s.t. gi(x) <= 0, i=1,...,m  (inequality constraints)
         hj(x) = 0, j = 1,..., n (equality constraints)

    Parameters
    ----------
    fun : CFunction
        The objective function to be optimized,
        along with 1st-order (Jacobian) and 2nd-order (Hessian) derivatives
        (if available).

    """
    __metaclass__ = ABCMeta
    __super__ = 'COptimizer'

    def __init__(self, fun):
        """Class initializer."""
        if not isinstance(fun, CFunction):
            raise TypeError('Input expected to be `CFunction`')
        self._fun = fun

    @property
    def fun(self):
        """Function to optimize."""
        return self._fun

    @abstractmethod
    def minimize(self, x0, *args, **kwargs):
        """Minimize function."""
        raise NotImplementedError("Minimize not implemented!")
