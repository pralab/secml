"""
.. module:: CSolverScipy
   :synopsis: Interface for function optimization and minimization

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>
.. moduleauthor:: Battista Biggio <battista.biggio@diee.unica.it>

"""
from scipy import optimize as sc_opt

from secml.array import CArray
from secml.optimization.optimizers import COptimizer


class CSolverScipy(COptimizer):
    """Implements optimizers from scipy."""
    __class_type = 'scipy'

    def minimize(self, x0, *args, **kwargs):
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
        >>> from secml.optimization.optimizers import CSolverScipy
        >>> from secml.optimization.function import CFunctionRosenbrock

        >>> x0 = CArray([1.3, 0.7])
        >>> opt = CSolverScipy(CFunctionRosenbrock())
        >>> min_x, jac, fun_val, res = opt.minimize(
        ... x0, method='BFGS', options={'gtol': 1e-6, 'disp': True})
        Optimization terminated successfully.
                 Current function value: 0.000000
                 Iterations: 32
                 Function evaluations: 39
                 Gradient evaluations: 39
        >>> print min_x
        CArray([1. 1.])
        >>> print jac
        CArray([ 3.230858e-08 -1.558678e-08])
        >>> print fun_val
        9.29438398164e-19
        >>> print res['message']
        Optimization terminated successfully.

        """
        if x0.issparse is True or x0.is_vector_like is False:
            raise ValueError("x0 must be a dense flat array")

        # passing jac to scipy minimize
        jac = kwargs['jac'] if 'jac' in kwargs else self.fun.gradient_ndarray
        kwargs['jac'] = jac

        sc_opt_out = sc_opt.minimize(self.fun.fun_ndarray,
                                     x0.ravel().tondarray(),
                                     *args, **kwargs)

        sc_opt_out_msg = {'status': sc_opt_out.status,
                          'success': sc_opt_out.success,
                          'message': sc_opt_out.message,
                          'nfev': sc_opt_out.nfev, 'nit': sc_opt_out.nit}

        return CArray(sc_opt_out.x), CArray(sc_opt_out.jac), \
            sc_opt_out.fun, sc_opt_out_msg
