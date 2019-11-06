"""
.. module:: COptimizerScipy
   :synopsis: Interface for function optimization and minimization

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from scipy import optimize as sc_opt

from secml.array import CArray
from secml.optim.optimizers import COptimizer

SUPPORTED_METHODS = ['BFGS', 'L-BFGS-B']


class COptimizerScipy(COptimizer):
    """Implements optimizers from scipy.

    Attributes
    ----------
    class_type : 'scipy-opt'

    """
    __class_type = 'scipy-opt'

    def _bounds_to_scipy(self):
        """Converts bounds to scipy format.

        Returns
        -------
        scipy.optimize.Bounds
            Bounds constraint in scipy-compatible format.

        """
        if self.bounds is None:
            return None

        # Scalar or CArray
        lb = self.bounds.lb
        ub = self.bounds.ub

        # If bounds are vectors, transform to ndarray
        lb = lb.tondarray() if isinstance(lb, CArray) else lb
        ub = ub.tondarray() if isinstance(ub, CArray) else ub

        # return scipy bounds
        return sc_opt.Bounds(lb, ub)

    def minimize(self, x_init, args=(), **kwargs):
        """Minimize function.

        Wrapper of `scipy.optimize.minimize`.

        Parameters
        ----------
        x_init : CArray
            Init point. Dense flat array of real elements of size 'n',
            where 'n' is the number of independent variables.
        args : tuple, optional
            Extra arguments passed to the objective function and its
            derivatives (`fun`, `jac` and `hess` functions).

        The following can be passed as optional keyword arguments:

        method : str or callable, optional
            Type of solver.  Should be one of

              - 'BFGS'        :ref:`(see here) <optimize.minimize-bfgs>`
              - 'L-BFGS-B'    :ref:`(see here) <optimize.minimize-lbfgsb>`

            If not given, chosen to be one of ``BFGS`` or ``L-BFGS-B``
            depending if the problem has constraints or bounds.
            See `c_optimizer_scipy.SUPPORTED_METHODS` for the full list.
        jac : {'2-point', '3-point', 'cs', bool}, optional
            Method for computing the gradient vector.
            The function in `self.fun.gradient` will be used (if defined).
            Alternatively, the keywords {'2-point', '3-point', 'cs'} select a
            finite difference scheme for numerical estimation of the gradient.
            Options '3-point' and 'cs' are available only to 'trust-constr'.
            If `jac` is a Boolean and is True, `fun` is assumed to return the
            gradient along with the objective function. If False, the gradient
            will be estimated using '2-point' finite difference estimation.
        bounds : scipy.optimize.Bounds, optional
            A bound constraint in scipy.optimize format. If defined, bounds
            of `COptimizerScipy` will be ignored.
        tol : float, optional
            Tolerance for termination. For detailed control,
            use solver-specific options.
        options : dict, optional
            A dictionary of solver options. All methods accept the following
            generic options:
            
             - maxiter : int
               Maximum number of iterations to perform.
             - disp : bool
               Set to True to print convergence messages.
               Equivalent of setting `COptimizerScipy.verbose = 2`.

            For method-specific options, see :func:`show_options()`.

        Returns
        -------
        x : CArray
            The solution of the optimization.

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
        >>> print(x_opt)
        CArray([1. 1.])
        >>> print(opt.f_opt)
        9.294383981640425e-19

        """
        if x_init.issparse is True or x_init.is_vector_like is False:
            raise ValueError("x0 must be a dense flat array")

        # reset fun and grad eval counts for both fun and f (by default fun==f)
        self._f.reset_eval()
        self._fun.reset_eval()

        # select method
        method = kwargs['method'] if 'method' in kwargs else None
        if method is None:
            # Only 'L-BFGS-B` supports bounds
            method = 'BFGS' if self.bounds is None else 'L-BFGS-B'
        # check if method is supported
        if method not in SUPPORTED_METHODS:
            raise NotImplementedError("selected method is not supported.")
        # set method
        kwargs['method'] = method

        # we're not supporting any solver with constraints at this stage
        if self.constr is not None:
            raise NotImplementedError("constraints are not supported.")

        # converting input parameters to scipy
        # 1) gradient (jac)
        jac = kwargs['jac'] if 'jac' in kwargs else self._fun.gradient_ndarray
        kwargs['jac'] = jac
        # 2) bounds
        bounds = kwargs['bounds'] if 'bounds' in kwargs else None
        if bounds is None:
            bounds = self._bounds_to_scipy()
        kwargs['bounds'] = bounds

        if self.verbose >= 2:  # Override verbosity options
            kwargs['options']['disp'] = True

        # call minimize now
        sc_opt_out = sc_opt.minimize(self._fun.fun_ndarray,
                                     x_init.ravel().tondarray(),
                                     args=args, **kwargs)

        if not sc_opt_out.success:
            self.logger.warning(
                "Optimization has not exited successfully!\n")

        if self.verbose >= 1:
            # Workaround for scipy message randomly being a str or bytes
            if isinstance(sc_opt_out.message, str):
                self.logger.info(sc_opt_out.message + "\n")
            else:
                self.logger.info(str(sc_opt_out.message, 'ascii') + "\n")

        self._f_seq = CArray(sc_opt_out.fun)  # only last iter available

        self._x_opt = CArray(sc_opt_out.x)

        # check if point is valid
        # i.e., if the selected solver does not ignore the constraints
        if self.constr is not None and self.constr.is_violated(self.x_opt):
            self.logger.warning("Constraints are not satisfied. "
                                "The scipy solver may be ignoring them.\n")

        if self.bounds is not None and self.bounds.is_violated(self.x_opt):
            self.logger.warning("Bounds are not satisfied. "
                                "The scipy solver may be ignoring them.\n")

        return self.x_opt
