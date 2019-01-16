"""
.. module:: CSolver
   :synopsis: Abstract class providing a common interface
    to implement inherited evasion solvers
    (available now: only descent_direction).

    @author: Battista Biggio

"""
from abc import ABCMeta, abstractmethod, abstractproperty

from secml.core import CCreator
from secml.optimization.function import CFunction
from secml.optimization.constraints import CConstraint


class CSolver(CCreator):
    """
    This is an abstract class for optimizing:

        min  f(x)
        s.t. d(x,x0) <= dmax
             x_lb <= x <= x_ub

    f(x) is the objective function (either linear or nonlinear),
    d(x,x0) <= dmax is a distance constraint in feature space (l1 or l2),
    and x_lb <= x <= x_ub is a box constraint on x.

    The solution algorithm is based on a line-search exploring one feature
    (i.e., dimension) at a time (for l1-constrained problems), or all features
    (for l2-constrained problems). This solver also works for discrete
    problems, where x is integer valued. In this case, exploration works
    by manipulating one feature at a time.

    """
    __metaclass__ = ABCMeta
    __super__ = 'CSolver'

    def __init__(self,
                 fun,
                 constr=None,
                 bounds=None,
                 discrete=False):

        # this is the function passed by the user to be maximized or minimized
        self._f = None
        self._constr = None
        self._bounds = None
        self._discrete = None

        # this is the internal function to be always minimized
        self._fun = None  # by default, minimize f(x), so fun=f

        # calling setters to check types
        self.f = fun  # this will set both f and fun
        self.constr = constr
        self.bounds = bounds
        self.discrete = discrete

        CSolver.__clear(self)

    ##########################################
    #            INTERNALS
    ##########################################
    def __clear(self):
        """Reset the object."""
        if self._f is not None:
            self._f.clear()
        if self._fun is not None:
            self._fun.clear()
        if self.constr is not None:
            self.constr.clear()
        if self.bounds is not None:
            self.bounds.clear()

        self._x_opt = None  # solution point
        self._f_opt = None  # last score f_seq[-1]
        self._f_seq = None  # sequence of fun values at each iteration
        self._x_seq = None  # sequence of x values at each iteration
        self._f_eval = 0
        self._grad_eval = 0

    def __is_clear(self):
        """Returns True if object is clear."""
        if self._f is not None and not self._f.is_clear():
            return False
        if self._fun is not None and not self._fun.is_clear():
            return False
        if self.constr is not None and not self.constr.is_clear():
            return False
        if self.bounds is not None and not self.bounds.is_clear():
            return False

        if self._x_opt is not None or self._f_opt is not None:
            return False
        if self._f_seq is not None or self._x_seq is not None:
            return False

        if self._f_eval + self._grad_eval != 0:
            return False

        return True

    ###########################################################################
    #                           READ-ONLY ATTRIBUTES
    ###########################################################################
    @property
    def n_dim(self):
        return int(self._fun.n_dim)

    @property
    def x_opt(self):
        return self._x_opt

    @property
    def f_opt(self):
        return self._f_seq[-1].item()

    @property
    def x_seq(self):
        return self._x_seq

    @property
    def f_seq(self):
        return self._f_seq

    @property
    def f_eval(self):
        return self._f_eval

    @property
    def grad_eval(self):
        return self._grad_eval

    ###########################################################################
    #                           READ-WRITE ATTRIBUTES
    ###########################################################################

    @property
    def f(self):
        """Returns objective function"""
        return self._f

    @f.setter
    def f(self, f):
        if not isinstance(f, CFunction):
            raise TypeError(
                "Input parameter is not a `CFunction` object.")
        self._f = f
        self._fun = f
        # changing optimization problem requires clearing the solver
        self.__clear()

    @property
    def constr(self):
        return self._constr

    @constr.setter
    def constr(self, constr):

        # constr is optional
        if constr is None:
            self._constr = None
            self.__clear()
            return

        if not isinstance(constr, CConstraint):
            raise TypeError(
                "Input parameter is not a `CConstraint` object.")

        if constr.class_type != 'l1' and constr.class_type != 'l2':
            raise TypeError(
                "Only l1 or l2 `CConstraint` objects are accepted as input.")

        self._constr = constr
        # changing optimization problem requires clearing the solver
        self.__clear()

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, bounds):

        # bounds is optional
        if bounds is None:
            self._bounds = None
            self.__clear()
            return

        if not isinstance(bounds, CConstraint):
            raise TypeError(
                "Input parameter is not a `CConstraint` object.")

        if bounds.class_type != 'box':
            raise TypeError(
                "Only box `CConstraint` objects are accepted as input.")

        self._bounds = bounds
        # changing optimization problem requires clearing the solver
        self.__clear()

    @property
    def discrete(self):
        """Returns True if feature space is discrete, False if continuous."""
        return self._discrete

    @discrete.setter
    def discrete(self, value):
        """Set to True if feature space is discrete, False if continuous."""
        self._discrete = bool(value)
        # changing optimization problem requires clearing the solver
        self.__clear()

    ##########################################
    #            PUBLIC METHODS
    ##########################################

    @abstractmethod
    def minimize(self, x_init=None):
        raise NotImplementedError('Function `minimize` is not implemented.')

    def maximize(self, x_init=None):

        # invert sign of fun(x) and grad(x) and run minimize
        self._fun = CFunction(
            fun=lambda z: -self._f.fun(z),
            gradient=lambda z: -self._f.gradient(z))

        x = self.minimize(x_init)
        # fix solution variables
        self._f_seq = - self._f_seq

        # restore fun to its default
        self._fun = self.f

        return x
