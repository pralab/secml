"""
.. module:: COptimizer
   :synopsis: Interface for function optimization and minimization.

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>

"""
from abc import ABCMeta, abstractmethod
from functools import partial

from secml.core import CCreator
from secml.optim.function import CFunction
from secml.optim.constraints import CConstraint, CConstraintBox


class COptimizer(CCreator, metaclass=ABCMeta):
    """Interface for optimizers.

    Implements:
     minimize f(x)
     s.t. gi(x) <= 0, i=1,...,m  (inequality constraints)
     hj(x) = 0, j = 1,..., n (equality constraints)

    Parameters
    ----------
    fun : CFunction
        The objective function to be optimized, along with 1st-order (Jacobian)
        and 2nd-order (Hessian) derivatives (if available).
    constr : CConstraintL1 or CConstraintL2 or None, optional
        A distance constraint. Default None.
    bounds : CConstraintBox or None, optional
        A box constraint. Default None.

    """
    __super__ = 'COptimizer'

    def __init__(self, fun, constr=None, bounds=None):

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

        if not isinstance(bounds, CConstraintBox):
            raise TypeError(
                "Input parameter is not a `CConstraintBox` object.")

        self._bounds = bounds

    ##########################################
    #                METHODS
    ##########################################

    @abstractmethod
    def minimize(self, x_init, args=(), **kwargs):
        """Interface for minimizers.

        Implementing:
            min fun(x)
            s.t. constraint

        Parameters
        ----------
        x_init : CArray
            The initial input point.
        args : tuple, optional
            Extra arguments passed to the objective function and its gradient.
        kwargs
            Additional parameters of the minimization method.

        """
        raise NotImplementedError('Function `minimize` is not implemented.')

    def maximize(self, x_init, args=(), **kwargs):
        """Interface for maximizers.

        Implementing:
            max fun(x)
            s.t. constraint

        This is implemented by inverting the sign of fun and gradient and
        running the `COptimizer.minimize()`.

        Parameters
        ----------
        x_init : CArray
            The initial input point.
        args : tuple, optional
            Extra arguments passed to the objective function and its gradient.
        kwargs
            Additional parameters of the minimization method.

        """
        # Invert sign of fun(x) and grad(x) and run minimize
        # We use def statements and partial to respect PEP8 and scopes

        def fun_inv(wrapped_fun, z, *f_args, **f_kwargs):
            return -wrapped_fun(z, *f_args, **f_kwargs)

        def grad_inv(wrapped_grad, z, *f_args, **f_kwargs):
            return -wrapped_grad(z, *f_args, **f_kwargs)

        self._fun = CFunction(
            fun=partial(fun_inv, self._f.fun),
            gradient=partial(grad_inv, self._f.gradient)
        )

        x = self.minimize(x_init, args=args, **kwargs)

        # fix solution variables
        self._f_seq = -self._f_seq

        # restore fun to its default
        self._fun = self.f

        return x
