"""
.. module:: COptimizerScipy
   :synopsis: Interface for function optimization and minimization

.. moduleauthor:: Battista Biggio <battista.biggio@diee.unica.it>
.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from scipy import optimize as sc_opt

from secml.array import CArray
from secml.optim.optimizers import COptimizer


class COptimizerScipy(COptimizer):
    """Implements optimizers from scipy.

    Attributes
    ----------
    class_type : 'scipy-opt'

    """
    __class_type = 'scipy-opt'

    def minimize(self, x_init, *args, **kwargs):
        """Minimize function.

        Wrapper of `scipy.optimize.minimize`.

        Parameters
        ----------
        x_init : CArray
            Init point. Dense flat array of real elements of size 'n',
            where 'n' is the number of independent variables.
        args : any, optional
            Extra arguments passed to the objective function and its
            derivatives (`fun`, `jac` and `hess` functions).

        The following can be passed as optional keyword arguments:

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
                    Equivalent of setting `COptimizerScipy.verbose = 2`.
            For method-specific options, see :func:`show_options()`.

        Returns
        -------
        x : CArray
            The solution of the optimization.

        Warnings
        --------
        Due to limitations of the current wrappers,
        not all solver methods listed above are supported.

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.optim.optimizers import COptimizerScipy
        >>> from secml.optim.function import CFunctionRosenbrock

        >>> x_init = CArray([1.3, 0.7])
        >>> opt = COptimizerScipy(CFunctionRosenbrock())
        >>> x_opt = opt.minimize(
        ... x_init, method='BFGS', options={'gtol': 1e-6, 'disp': True})
        Optimization terminated successfully.
                 Current function value: 0.000000
                 Iterations: 32
                 Function evaluations: 39
                 Gradient evaluations: 39
        >>> print x_opt
        CArray([1. 1.])
        >>> print opt.f_opt
        9.29438398164e-19

        """
        if x_init.issparse is True or x_init.is_vector_like is False:
            raise ValueError("x0 must be a dense flat array")

        # converting input parameters to scipy
        # 1) jac
        jac = kwargs['jac'] if 'jac' in kwargs else self.f.gradient_ndarray
        kwargs['jac'] = jac

        # 2) bounds
        bounds = kwargs['bounds'] if 'bounds' in kwargs else None
        if bounds is None and self.bounds is not None:
            # set our bounds
            bounds = sc_opt.Bounds(self.bounds.lb.tondarray(),
                                   self.bounds.ub.tondarray())
        kwargs['bounds'] = bounds

        if self.verbose >= 2:  # Override verbosity options
            kwargs['options']['disp'] = True

        # call minimize now
        sc_opt_out = sc_opt.minimize(self.f.fun_ndarray,
                                     x_init.ravel().tondarray(),
                                     args=args, **kwargs)

        if sc_opt_out.status != 0:
            self.logger.warning(
                "Optimization has not terminated successfully!")
        elif self.verbose >= 1:
            self.logger.info(sc_opt_out.message)

        self._f_seq = CArray(sc_opt_out.fun)  # only last iter available

        return CArray(sc_opt_out.x)
