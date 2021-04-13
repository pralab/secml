"""
.. module:: COptimizerPGD
   :synopsis: Optimizer using Projected Gradient Descent

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from secml.array import CArray
from secml.optim.optimizers import COptimizer


class COptimizerPGD(COptimizer):
    """Solves the following problem:

    min  f(x)
    s.t. d(x,x0) <= dmax
    x_lb <= x <= x_ub

    f(x) is the objective function (either linear or nonlinear),
    d(x,x0) <= dmax is a distance constraint in feature space (l1 or l2),
    and x_lb <= x <= x_ub is a box constraint on x.

    The solution algorithm is based on the classic gradient descent algorithm.

    Parameters
    ----------
    fun : CFunction
        The objective function to be optimized, along with 1st-order (Jacobian)
        and 2nd-order (Hessian) derivatives (if available).
    constr : CConstraintL1 or CConstraintL2 or None, optional
        A distance constraint. Default None.
    bounds : CConstraintBox or None, optional
        A box constraint. Default None.
    eta : scalar, optional
        Step of the Projected Gradient Descent. Default 1e-3.
    eps : scalar, optional
        Tolerance of the stop criterion. Default 1e-4.
    max_iter : int, optional
        Maximum number of iterations. Default 200.

    Attributes
    ----------
    class_type : 'pgd'

    """
    __class_type = 'pgd'

    def __init__(self, fun,
                 constr=None,
                 bounds=None,
                 eta=1e-3,
                 eps=1e-4,
                 max_iter=200):

        COptimizer.__init__(self, fun=fun,
                            constr=constr, bounds=bounds)

        # Read/write attributes
        self.eta = eta  # gradient step size
        self.max_iter = max_iter  # maximum number of iterations
        self.eps = eps  # tolerance value for stop criterion

    ###########################################################################
    #                        READ/WRITE ATTRIBUTES
    ###########################################################################

    @property
    def eta(self):
        """Return gradient descent step"""
        return self._eta

    @eta.setter
    def eta(self, value):
        """Set gradient descent step"""
        self._eta = float(value)

    @property
    def max_iter(self):
        """Returns the maximum number of gradient descent iteration"""
        return self._max_iter

    @max_iter.setter
    def max_iter(self, value):
        """Set the maximum number of gradient descent iteration"""
        self._max_iter = int(value)

    @property
    def eps(self):
        """Return tolerance value for stop criterion"""
        return self._eps

    @eps.setter
    def eps(self, value):
        """Set tolerance value for stop criterion"""
        self._eps = float(value)

    #############################################
    #                  METHODS
    #############################################

    def _return_best_solution(self, i):
        """Return the best solution among the ones found up to iteration i.

        Parameters
        ----------
        i : int
            Index of the current iteration.

        Returns
        -------
        x_opt : CArray
            Best point found so far.

        """
        f_seq = self.f_seq[:i]
        best_sol_idx = f_seq.argmin()

        self._x_seq = self.x_seq[:best_sol_idx + 1, :]
        self._f_seq = self.f_seq[:best_sol_idx + 1]
        self._x_opt = self._x_seq[-1, :]

        return self._x_opt

    def minimize(self, x_init, args=(), **kwargs):
        """Interface to minimizers.

        Implements:
            min fun(x)
            s.t. constraint

        Parameters
        ----------
        x_init : CArray
            The initial input point.
        args : tuple, optional
            Extra arguments passed to the objective function and its gradient.

        Returns
        -------
        f_seq : CArray
            Array containing values of f during optimization.
        x_seq : CArray
            Array containing values of x during optimization.

        """
        if len(kwargs) != 0:
            raise ValueError(
                "{:} does not accept additional parameters.".format(
                    self.__class__.__name__))

        # reset fun and grad eval counts for both fun and f (by default fun==f)
        self._f.reset_eval()
        self._fun.reset_eval()

        # constr.radius = 0, exit
        if self.constr is not None and self.constr.radius == 0:
            # classify x0 and return
            x0 = self.constr.center
            if self.bounds is not None and self.bounds.is_violated(x0):
                import warnings
                warnings.warn(
                    "x0 " + str(x0) + " is outside of the given bounds.",
                    category=RuntimeWarning)
            self._x_seq = CArray.zeros((1, x0.size),
                                       sparse=x0.issparse, dtype=x0.dtype)
            self._f_seq = CArray.zeros(1)
            self._x_seq[0, :] = x0
            self._f_seq[0] = self._fun.fun(x0, *args)
            self._x_opt = x0
            return x0

        # if x is outside of the feasible domain, project it
        if self.bounds is not None and self.bounds.is_violated(x_init):
            x_init = self.bounds.projection(x_init)

        if self.constr is not None and self.constr.is_violated(x_init):
            x_init = self.constr.projection(x_init)

        if (self.bounds is not None and self.bounds.is_violated(x_init)) or \
                (self.constr is not None and self.constr.is_violated(x_init)):
            raise ValueError(
                "x_init " + str(x_init) + " is outside of feasible domain.")

        self._x_seq = CArray.zeros(
            (self._max_iter, x_init.size), sparse=x_init.issparse)
        self._f_seq = CArray.zeros(self._max_iter)

        x = x_init.deepcopy()

        i = 0
        for i in range(self._max_iter):

            self._x_seq[i, :] = x
            self._f_seq[i] = self._fun.fun(x, *args)

            if i > 0 and abs(self.f_seq[i - 1] - self.f_seq[i]) < self.eps:
                self.logger.debug("Flat region, exiting... {:}  {:}".format(
                    self._f_seq[i], self._f_seq[i - 1]))
                return self._return_best_solution(i)

            if i > 10 and abs(self.f_seq[i - 5:i].mean() -
                              self.f_seq[i - 10:i - 5].mean()) < self.eps:
                self.logger.debug(
                    "Flat region over 10 iterations, exiting... {:}  {:}".format(
                        self.f_seq[i - 3:i].mean(),
                        self.f_seq[i - 6:i - 3].mean()))
                return self._return_best_solution(i)

            grad = self._fun.gradient(x, *args)

            # debugging information
            self.logger.debug(
                'Iter.: ' + str(i) + ', f(x): ' +
                str(self._f_seq[i].item()) + ', |df/dx|: ' + str(grad.norm()))

            # make a step into the deepest descent direction
            x -= self.eta * grad

            # project x onto the feasible domain
            if self.constr is not None and self.constr.is_violated(x):
                x = self.constr.projection(x)
            if self.bounds is not None and self.bounds.is_violated(x):
                x = self.bounds.projection(x)

        return self._return_best_solution(i)
