"""
.. module:: COptimizer
   :synopsis: Interface for function optimization and minimization.

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>
.. moduleauthor:: Battista Biggio <battista.biggio@diee.unica.it>

"""
from abc import ABCMeta, abstractmethod
import six

from secml.core import CCreator
from secml.optim.function import CFunction
from secml.optim.constraints import CConstraint


@six.add_metaclass(ABCMeta)
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
    __super__ = 'COptimizer'

    def __init__(self, fun, constr=None, bounds=None, discrete=False):

        # The following will set both f and fun
        # fun: the internal function to be always minimized
        # f: the "public" function. By default, minimize f(x), so fun=f
        if not isinstance(fun, CFunction):
            raise TypeError(
                "Input parameter is not a `CFunction` object.")
        self._f = fun
        self._fun = fun

        # Read/write attributes
        self.constr = constr
        self.bounds = bounds
        self.discrete = discrete

        # Internal attributes
        self._x_opt = None  # solution point
        self._f_opt = None  # last score f_seq[-1]
        self._f_seq = None  # sequence of fun values at each iteration
        self._x_seq = None  # sequence of x values at each iteration

    ###########################################################################
    #                           READ-ONLY ATTRIBUTES
    ###########################################################################

    @property
    def f(self):
        """The objective function"""
        return self._f

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
    def n_dim(self):
        return self._fun.n_dim

    @property
    def f_eval(self):
        return self._fun.n_fun_eval

    @property
    def grad_eval(self):
        return self._fun.n_grad_eval

    ###########################################################################
    #                           READ-WRITE ATTRIBUTES
    ###########################################################################

    @property
    def constr(self):
        """Optimization constraint."""
        return self._constr

    @constr.setter
    def constr(self, constr):
        """Optimization constraint."""
        if constr is None:
            self._constr = None
            return

        if not isinstance(constr, CConstraint):
            raise TypeError(
                "Input parameter is not a `CConstraint` object.")

        if constr.class_type != 'l1' and constr.class_type != 'l2':
            raise TypeError(
                "Only l1 or l2 `CConstraint` objects are accepted as input.")

        self._constr = constr

    @property
    def bounds(self):
        """Optimization bounds."""
        return self._bounds

    @bounds.setter
    def bounds(self, bounds):
        """Optimization bounds."""
        if bounds is None:
            self._bounds = None
            return

        if not isinstance(bounds, CConstraint):
            raise TypeError(
                "Input parameter is not a `CConstraint` object.")

        if bounds.class_type != 'box':
            raise TypeError(
                "Only box `CConstraint` objects are accepted as input.")

        self._bounds = bounds

    @property
    def discrete(self):
        """True if feature space is discrete, False if continuous."""
        return self._discrete

    @discrete.setter
    def discrete(self, value):
        """True if feature space is discrete, False if continuous."""
        self._discrete = bool(value)

    ##########################################
    #                METHODS
    ##########################################

    @abstractmethod
    def minimize(self, x_init, *args, **kwargs):
        # TODO: ADD DOCSTRING
        raise NotImplementedError('Function `minimize` is not implemented.')

    def maximize(self, x_init, *args, **kwargs):
        # TODO: ADD DOCSTRING

        # invert sign of fun(x) and grad(x) and run minimize
        self._fun = CFunction(
            fun=lambda z: -self._f.fun(z, *args),
            gradient=lambda z: -self._f.gradient(z, *args))

        x = self.minimize(x_init, *args, **kwargs)

        # fix solution variables
        self._f_seq = -self._f_seq

        # restore fun to its default
        self._fun = self.f

        return x
