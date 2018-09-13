"""
.. module:: OptimizerOpenOpt
   :synopsis: Interface for function optimization and minimization with OpenOpt

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>
.. moduleauthor:: Davide Maiorca <davide.maiorca@diee.unica.it>
.. moduleauthor:: Paolo Russu <paolo.russu@diee.unica.it>

"""
from abc import ABCMeta, abstractmethod

from prlib.core import CCreator
from prlib.optimization.function import CFunction
from prlib.array import CArray
from scipy import optimize as sc_opt
from prlib.core.constants import eps

class COptimizer(CCreator):
    """Abstract class for implementing optimizers."""
    __metaclass__ = ABCMeta
    __super__ = 'COptimizer'

    def __init__(self, fun, solver=None):

        if not isinstance(fun, CFunction):
            raise TypeError("Input parameter is not a `CFunction`.")

        self.fun = fun
        self.solver = solver

        COptimizer.__clear(self)

    def __clear(self):
        self._x_star = None
        self._score = None
        self._solver_output = None

    @property
    def fun(self):
        """Function or vector to optimize."""
        return self._fun

    @fun.setter
    def fun(self, val):
        """Function or vector to optimize."""
        if not isinstance(val, CFunction):
            raise TypeError('Input expected to be `CFunction`')
        self._fun = val

    @property
    def solver(self):
        return self._solver

    @solver.setter
    def solver(self, val):
        """
        val : str
            Identifier of the solver to use. Default `pclp`.
            A faster alternative is `lpSolve` (needs separate installation).
        """
        if val is None or isinstance(val, basestring):
            self._solver = val
        else:
            raise TypeError('Input expected to be None or a string \
                        specifying a valid solver')

    @property
    def n_dim(self):
        """Returns the dimensionality of x (input of fun)."""
        return self._fun.n_dim

    # TODO: questa cosa e' da sistemare.
    # in questa classe non servono cose
    # specifiche di OpenOpt
    def _store_result(self, result):
        """Sets the final point and the score."""
        # Final optimizer point
        self._x_star = CArray(result.xf)[:self.n_dim]
        # Final objective function value
        self._score = result.ff
        # Other useful solver output
        self._solver_output = {'slack': CArray(result.xf)[self.n_dim:] if self.n_dim < len(result.xf) else None,
                               'isFeasible': result.isFeasible,
                               'solver_name': result.solverInfo['name'],
                               'solver_time': result.elapsed['solver_time'],
                               # TODO: MAP stopcase TO SOMETHING READABLE
                               'solver_stopcase': result.stopcase
                               }

#     @abstractmethod
#     def minimize(self, x=None):
#         raise NotImplementedError(
#             "`minimize` method must be implemented by {:} subclasses."
#             "".format(COptimizer.__name__))


    @staticmethod
    def _fun_toarray(xk, fun, fprime, *args):
        """Wrapper for functions that use and return CArrays.

        This function wraps any callable (bound) method that
        takes as input CArray(s) and return CArray(s) in order
        to take any array like object and return an ndarray.

        Even if both fun and fprime are required, this function
        runs `fun` only. See :meth:`_fprime_toarray` for `fprime`
        wrapper.

        Parameters
        ----------
        xk : array_like
            First input of the function.
        fun : bound method
            Callable function that takes CArray(s) as input and
            return a CArray as result.
        fprime : bound method
            Callable function that takes CArray(s) as input and
            return a CArray as result.
        args : \*args, optional
            Extra arguments passed to `fun`.

        Returns
        -------
        out_fun : ndarray
            Output of 'fun' casted to ndarray.

        """
        # NOTE: fprime is not used. This is a wrapper for fun
        return CArray(fun(CArray(xk), *args)).tondarray()



    @staticmethod
    def _fprime_toarray(xk, fun, fprime, *args):
        """Wrapper for functions that use and return CArrays.

        This function wraps any callable (bound) method that
        takes as input CArray(s) and return CArray(s) in order
        to take any array like object and return an ndarray.

        Even if both fun and fprime are required, this function
        runs `fprime` only. See :meth:`_fun_toarray` for `fun`
        wrapper.

        Parameters
        ----------
        xk : array_like
            First input of the function.
        fun : bound method
            Callable function that takes CArray(s) as input and
            return a CArray as result.
        fprime : bound method
            Callable function that takes CArray(s) as input and
            return a CArray as result.
        args : \*args, optional
            Extra arguments passed to `fun`.

        Returns
        -------
        out_fprime : ndarray
            Output of 'fun' casted to ndarray.

        """
        # NOTE: fun is not used. This is a wrapper for fprime
        return CArray(fprime(CArray(xk), *args)).tondarray()


    def approx_fprime(self, xk, epsilon, *args):
        """Finite-difference approximation of the gradient of a
        scalar function.

        Wrapper for scipy function :func:`scipy.optimize.approx_fprime`.

        Parameters
        ----------
        xk : CArray
            Flat array with pattern at which to determine the gradient of `f`.
        epsilon : scalar or CArray
            Increment of `xk` to use for determining the function gradient.
            If a scalar, uses the same finite difference delta for all partial
            derivatives.  If an array, should contain one value per element of
            `xk`. A default value is provided by `COptimizer.epsilon` but should
            be passed explicitly to `approx_fprime`.
        \*args : args, optional
            Any other arguments that are to be passed to `fun`.

        Returns
        -------
        grad : CArray
            The partial derivatives of `fun` to `xk`.

        See Also
        --------
        .check_grad : Check correctness of gradient function against approx_fprime.

        Notes
        -----
        The function gradient is determined by the forward finite difference
        formula::

                     fun(xk[i] + epsilon[i]) - f(xk[i])
           fun'[i] = -----------------------------------
                                epsilon[i]

        The main use of `approx_fprime` is to determine numerically
        the Jacobian of a function.

        Examples
        --------
        >>> from prlib.array import CArray
        >>> from prlib.optimization import COptimizer

        >>> def func(x, c0, c1):
        ...     "Coordinate vector `x` should be an array of size two."
        ...     return c0 * x[0]**2 + c1*x[1]**2

        >>> c0, c1 = (1, 200)
        >>> eps = COptimizer.epsilon
        >>> COptimizer(func).approx_fprime(CArray.ones(2), [eps, (200 ** 0.5) * eps], c0, c1)
        CArray([   2.          400.00004198])

        """
        xk_ndarray = CArray(CArray(xk).ravel()).tondarray()  # double casting to always have a CArray
        epsilon = epsilon.tondarray() if isinstance(epsilon, CArray) else epsilon
        # We use fun_toarray as the main callable for scipy to have
        # always an ndarray as output of self.fun
        return CArray(sc_opt.approx_fprime(xk_ndarray, self._fun_toarray, epsilon, self.fun.fun, self.fun.gradient, *args))



    def check_grad(self, x, *args, **kwargs):
        """Check the correctness of a gradient function by comparing
        it against a (forward) finite-difference approximation of
        the gradient.

        Parameters
        ----------
        x : CArray
            Pattern to check function gradient against forward difference
            approximation of function gradient using `fun` stored in the
            COptimizer instance.
        epsilon : scalar or CArray
            Increment to `xk` to use for determining the function gradient.
            If a scalar, uses the same finite difference delta for all partial
            derivatives.  If an array, should contain one value per element of
            `xk`. If not provided, value of `COptimizer.epsilon` is used.
        args : \*args, optional
            Extra arguments passed to `fun` and `fprime`.

        Returns
        -------
        err : float
            The square root of the sum of squares (i.e. the l2-norm) of the
            difference between ``fprime(x, *args)`` and the finite difference
            approximation of `fprime` at the points `x` using `fun` stored in
            the COptimizer instance.

        Notes
        -----
        `epsilon` is the only keyword argument accepted by the function. Any
        other optional argument for `fun` and `fprime` should be passed as
        non-keyword.

        See Also
        --------
        .approx_fprime : Finite-difference approximation of the gradient of a scalar function.

        """
        # We now take 'epsilon' from kwargs and check if any other
        # input has been passed as kwargs (we do not want it)
        if 'epsilon' in kwargs :
            epsilon = kwargs.pop('epsilon', eps)
        else :
            epsilon = eps
        if kwargs:
            raise ValueError("Unknown keyword arguments: %r" % (list(kwargs.keys()),))
        x_carray = CArray(x)
        return CArray(self.fun.gradient(x_carray, *args) - self.approx_fprime(x_carray, epsilon, *args)).norm()

