"""
.. module:: Optimizer
   :synopsis: Interface for function optimization and minimization

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from abc import ABCMeta

from scipy import optimize as sc_opt

from secml.core import CCreator
from secml.array import CArray
from secml.optimization.function import CFunction
from secml.optimization.opt_utils import fun_tondarray, fprime_tondarray
from secml.core.constants import eps


class COptimizer(CCreator):
    """Generic optimizer.

    Parameters
    ----------
    fun : CFunction
        Function to be optimized.
    solver : or None, optional
        Solver to use for optimization.

    """
    __metaclass__ = ABCMeta
    __super__ = 'COptimizer'
    class_type = 'opt'

    def __init__(self, fun, solver=None):
        """Class initializer."""
        self.fun = fun
        self._solver = solver

    @property
    def fun(self):
        """Function to optimize."""
        return self._fun

    @fun.setter
    def fun(self, val):
        """Function to optimize.

        Parameters
        ----------
        val : CFunction
            Function to optimize.

        """
        if not isinstance(val, CFunction):
            raise TypeError('Input expected to be `CFunction`')

        self._fun = val

    @property
    def solver(self):
        """Solver to use for optimization."""
        return self._solver

    def minimize(self, x0, args=(), method=None, jac=None,
                 tol=None, options=None):
        """Minimize function.

        Wrapper of `scipy.optimize.minimize`.

        Parameters
        ----------
        x0 : CArray
            Initial guess. Dense flat array of real elements of size 'n',
            where 'n' is the number of independent variables.
        args : tuple, optional
            Extra arguments passed to the objective function and its
            derivatives (`fun`, `jac` and `hess` functions).
        method : str or callable, optional
            Type of solver.  Should be one of
                - 'Nelder-Mead' :ref:`(see here) <optimize.minimize-neldermead>`
                - 'Powell'      :ref:`(see here) <optimize.minimize-powell>`
                - 'CG'          :ref:`(see here) <optimize.minimize-cg>`
                - 'BFGS'        :ref:`(see here) <optimize.minimize-bfgs>`
                - 'Newton-CG'   :ref:`(see here) <optimize.minimize-newtoncg>`
                - 'L-BFGS-B'    :ref:`(see here) <optimize.minimize-lbfgsb>`
                - 'TNC'         :ref:`(see here) <optimize.minimize-tnc>`
                - 'COBYLA'      :ref:`(see here) <optimize.minimize-cobyla>`
                - 'SLSQP'       :ref:`(see here) <optimize.minimize-slsqp>`
                - 'trust-constr':ref:`(see here) <optimize.minimize-trustconstr>`
                - 'dogleg'      :ref:`(see here) <optimize.minimize-dogleg>`
                - 'trust-ncg'   :ref:`(see here) <optimize.minimize-trustncg>`
                - 'trust-exact' :ref:`(see here) <optimize.minimize-trustexact>`
                - 'trust-krylov' :ref:`(see here) <optimize.minimize-trustkrylov>`
                - custom - a callable object.
            If not given, chosen to be one of ``BFGS``, ``L-BFGS-B``, ``SLSQP``,
            depending if the problem has constraints or bounds.
        jac : {'2-point', '3-point', 'cs', bool}, optional
            Method for computing the gradient vector. Only for CG, BFGS,
            Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg, trust-krylov,
            trust-exact and trust-constr.
            The function in `self.fun.gradient` will be used (if defined).
            Alternatively, the keywords {'2-point', '3-point', 'cs'} select a
            finite difference scheme for numerical estimation of the gradient.
            Options '3-point' and 'cs' are available only to 'trust-constr'.
            If `jac` is a Boolean and is True, `fun` is assumed to return the
            gradient along with the objective function. If False, the gradient
            will be estimated using '2-point' finite difference estimation.
        tol : float, optional
            Tolerance for termination. For detailed control,
            use solver-specific options.
        options : dict, optional
            A dictionary of solver options. All methods accept the following
            generic options:
                maxiter : int
                    Maximum number of iterations to perform.
                disp : bool
                    Set to True to print convergence messages.
            For method-specific options, see :func:`show_options()`.

        Returns
        -------
        x : CArray
            The solution of the optimization.
        jac : CArray
            Value of the Jacobian.
        fun_val : scalar
            Value of the objective function.
        out_msg : dict
            Dictionary with other minimizer output.
            Refer to `scipy.optimize.OptimizeResult` description
            for more informations.

        Warnings
        --------
        Due to limitations of the current wrappers,
        not all solver methods listed above are supported.

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.optimization import COptimizer
        >>> from secml.optimization.function import CFunctionRosenbrock

        >>> x0 = CArray([1.3, 0.7])
        >>> opt = COptimizer(CFunctionRosenbrock())
        >>> min_x, jac, fun_val, res = opt.minimize(
        ... x0, method='BFGS', options={'gtol': 1e-6, 'disp': True})
        Optimization terminated successfully.
                 Current function value: 0.000000
                 Iterations: 32
                 Function evaluations: 39
                 Gradient evaluations: 39
        >>> print min_x
        CArray([ 1.  1.])
        >>> print jac
        CArray([  3.230858e-08  -1.558678e-08])
        >>> print fun_val
        9.29438398164e-19
        >>> print res['message']
        Optimization terminated successfully.

        """
        if x0.issparse is True or x0.is_vector_like is False:
            raise ValueError("x0 must be a dense flat array")

        # Manage any optional argument to be passed to fun
        if args is ():
            args = (self.fun.fun, self.fun.gradient)
        else:
            args = (self.fun.fun, self.fun.gradient, args)

        # This wraps fun.gradient to be usable with scipy.minimize
        jac = fprime_tondarray if jac is None else jac

        sc_opt_out = sc_opt.minimize(fun_tondarray, x0.ravel().tondarray(),
                                     args=args, method=method, jac=jac,
                                     tol=tol,  options=options)

        sc_opt_out_msg = {'status': sc_opt_out.status,
                          'success': sc_opt_out.success,
                          'message': sc_opt_out.message,
                          'nfev': sc_opt_out.nfev, 'nit': sc_opt_out.nit}

        return CArray(sc_opt_out.x), CArray(sc_opt_out.jac), \
            sc_opt_out.fun, sc_opt_out_msg

    def approx_fprime(self, xk, epsilon, *args):
        """Finite-difference approximation of the gradient of a scalar function.

        Wrapper for scipy function :func:`scipy.optimize.approx_fprime`.

        Parameters
        ----------
        xk : CArray
            The flat dense vector with the point at which to determine
            the gradient of `fun`.
        epsilon : scalar or CArray
            Increment of `xk` to use for determining the function gradient.
            If a scalar, uses the same finite difference delta for all partial
            derivatives.
            If an array, should contain one value per element of `xk`.
        *args : args, optional
            Any other arguments that are to be passed to `fun`.

        Returns
        -------
        grad : CArray
            The partial derivatives of `fun` to `xk`.

        See Also
        --------
        `.check_grad` : Check correctness of gradient function against `approx_fprime`.

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
        >>> from secml.array import CArray
        >>> from secml.optimization import COptimizer
        >>> from secml.optimization.function import CFunction
        >>> from secml.core.constants import eps

        >>> def func(x, c0, c1):
        ...     "Coordinate vector `x` should be an array of size two."
        ...     return c0 * x[0]**2 + c1*x[1]**2

        >>> c0, c1 = (1, 200)
        >>> COptimizer(CFunction(func)).approx_fprime(CArray.ones(2), [eps, (200 ** 0.5) * eps], c0, c1)
        CArray(2,)(dense: [   2.        400.000042])

        """
        if xk.issparse is True or xk.is_vector_like is False:
            raise ValueError("x0 must be a dense flat array")

        # double casting to always have a CArray
        xk_ndarray = CArray(xk).ravel().tondarray()

        epsilon = epsilon.tondarray() if isinstance(epsilon, CArray) else epsilon

        return CArray(
            sc_opt.approx_fprime(xk_ndarray, fun_tondarray, epsilon,
                                 self.fun.fun, self.fun.gradient, *args))

    def check_grad(self, x, *args, **epsilon):
        """Check the correctness of a gradient function by comparing
         it against a (forward) finite-difference approximation of
         the gradient.

        Parameters
        ----------
        x : CArray
            Flat dense pattern to check function gradient against
            forward difference approximation of function gradient.
        epsilon : scalar or CArray
            Increment to `xk` to use for determining the function gradient.
            If a scalar, uses the same finite difference delta for all partial
            derivatives.  If an array, should contain one value per element of
            `xk`. If not provided, value of `COptimizer.epsilon` is used.
        *args : *args, optional
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
        `.approx_fprime` : Finite-difference approximation of the gradient of a scalar function.

        Examples
        --------
        >>> from secml.optimization import COptimizer
        >>> from secml.optimization.function import CFunction

        >>> def func(x):
        ...     return x[0]**2 - 0.5 * x[1]**3
        >>> def grad(x):
        ...     return [2 * x[0], -1.5 * x[1]**2]

        >>> opt = COptimizer(CFunction(func, grad))
        >>> opt.check_grad(CArray([1.5, -1.5]))
        2.9802322387695312e-08

        """
        if x.issparse is True or x.is_vector_like is False:
            raise ValueError("x0 must be a dense flat array")

        # We now extract 'epsilon' if passed by the user
        if 'epsilon' in epsilon:
            epsilon = epsilon.pop('epsilon', eps)
        else:
            epsilon = eps

        # real value of the gradient on x
        grad = self.fun.gradient(x, *args)
        # value of the approximated gradient on x
        approx = self.approx_fprime(x, epsilon, *args)

        return (grad - approx).norm()
