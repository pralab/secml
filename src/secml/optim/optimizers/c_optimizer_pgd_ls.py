"""
.. module:: COptimizerPGDLS
   :synopsis: Optimizer using Projected Gradient Descent with Bisect Line Search.

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>

"""
import numpy as np

from secml.array import CArray
from secml.optim.optimizers import COptimizer
from secml.optim.optimizers.line_search import CLineSearchBisect


class COptimizerPGDLS(COptimizer):
    """Solves the following problem:

    min  f(x)
    s.t. d(x,x0) <= dmax
    x_lb <= x <= x_ub

    f(x) is the objective function (either linear or nonlinear),
    d(x,x0) <= dmax is a distance constraint in feature space (l1 or l2),
    and x_lb <= x <= x_ub is a box constraint on x.

    The solution algorithm is based on a line-search exploring one feature
    (i.e., dimension) at a time (for l1-constrained problems), or all features
    (for l2-constrained problems). This solver also works for discrete
    problems where x and the grid discretization (eta) are integer valued.

    Differently from standard line searches, it explores a subset of
    `n_dimensions` at a time. In this sense, it is an extension of the
    classical line-search approach.

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
        Minimum resolution of the line-search grid. Default 1e-3.
    eta_min : scalar or None, optional
        Initial step of the line search. Gets multiplied or divided by 2
        at each step until convergence. If None, will be set equal to eta.
        Default None.
    eta_max : scalar or None, optional
        Maximum step of the line search. Default None.
    max_iter : int, optional
        Maximum number of iterations. Default 1000.
    eps : scalar, optional
        Tolerance of the stop criterion. Default 1e-4.

    Attributes
    ----------
    class_type : 'pgd-ls'

    """
    __class_type = 'pgd-ls'

    def __init__(self, fun,
                 constr=None, bounds=None,
                 eta=1e-3,
                 eta_min=None,
                 eta_max=None,
                 max_iter=1000,
                 eps=1e-4):

        COptimizer.__init__(self, fun=fun,
                            constr=constr, bounds=bounds)

        # Read/write attributes
        self.eta = eta
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.max_iter = max_iter
        self.eps = eps

        # Internal attributes
        self._line_search = None
        self._dtype = None

    ###########################################################################
    #                           READ-WRITE ATTRIBUTES
    ###########################################################################

    @property
    def eta(self):
        return self._eta

    @eta.setter
    def eta(self, value):
        self._eta = value

    @property
    def eta_min(self):
        return self._eta_min

    @eta_min.setter
    def eta_min(self, value):
        self._eta_min = value

    @property
    def eta_max(self):
        return self._eta_max

    @eta_max.setter
    def eta_max(self, value):
        self._eta_max = value

    @property
    def max_iter(self):
        """Returns the maximum number of descent iterations"""
        return self._max_iter

    @max_iter.setter
    def max_iter(self, value):
        """Set the maximum number of descent iterations"""
        self._max_iter = int(value)

    @property
    def eps(self):
        """Return tolerance value for stop criterion"""
        return self._eps

    @eps.setter
    def eps(self, value):
        """Set tolerance value for stop criterion"""
        self._eps = float(value)

    ##########################################
    #                METHODS
    ##########################################

    def _init_line_search(self, eta, eta_min, eta_max):
        """Initialize line-search optimizer"""
        self._line_search = CLineSearchBisect(
            fun=self._fun,
            constr=self._constr,
            bounds=self._bounds,
            max_iter=50,
            eta=eta, eta_min=eta_min, eta_max=eta_max)

    @staticmethod
    def _l1_projected_gradient(grad):
        """
        Find v that maximizes v'grad onto the unary-norm l1 ball.
        This is the maximization of an inner product over the l1 ball,
        and the optimal (sparse) direction v is found by setting
        v = sign(grad) when abs(grad) is maximum and 0 elsewhere.
        """
        abs_grad = abs(grad)
        grad_max = abs_grad.max()
        # Get one of the features having grad_max value
        argmax_pos = CArray(abs_grad == grad_max).nnz_indices[1][:1]
        # TODO: not sure if proj_grad should be always sparse (grad is not)
        proj_grad = CArray.zeros(
            shape=grad.shape, sparse=grad.issparse, dtype=grad.dtype)
        proj_grad[argmax_pos] = grad[argmax_pos].sign()
        return proj_grad

    def _box_projected_gradient(self, x, grad):
        """
        Exclude from descent direction those features which,
        if modified according to the given descent direction,
        would violate the box constraint.

        """
        if self.bounds is None:
            return grad  # all features are feasible

        # x_lb and x_ub are feature manipulations that violate box
        # (the first vector is potentially sparse, so it has to be
        # the first argument of logical_and to avoid conversion to dense)
        # FIXME: the following condition is error-prone.
        #  Use (ad wrap in CArray) np.isclose with atol=1e-6, rtol=0
        # FIXME: converting grad to dense as the sparse vs sparse logical_and
        #  is too slow
        x_lb = (x.round(6) == CArray(self.bounds.lb).round(6)).logical_and(
            grad.todense() > 0).astype(bool)

        x_ub = (x.round(6) == CArray(self.bounds.ub).round(6)).logical_and(
            grad.todense() < 0).astype(bool)

        # reset gradient for unfeasible features
        grad[x_lb + x_ub] = 0
        return grad

    def _xk(self, x, fx, *args):
        """Returns a new point after gradient descent."""

        # compute gradient
        grad = self._fun.gradient(x, *args)
        self._grad = grad  # only used for visualization/convergence

        norm = grad.norm()
        if norm < 1e-20:
            return x, fx  # return same point (and exit optimization)

        grad = grad / norm

        # filter modifications that would violate bounds (to sparsify gradient)
        grad = self._box_projected_gradient(x, grad)

        if self.constr is not None and self.constr.class_type == 'l1':
            # project z onto l1 constraint (via dual norm)
            grad = self._l1_projected_gradient(grad)

        next_point = CArray(x - grad * self._line_search.eta,
                            dtype=self._dtype, tosparse=x.issparse)

        if self.constr is not None and self.constr.is_violated(next_point):
            self.logger.debug("Line-search on distance constraint.")
            grad = CArray(x - self.constr.projection(next_point))
            if self.constr.class_type == 'l1':
                grad = grad.sign()  # to move along the l1 ball surface
            z, fz = self._line_search.minimize(x, -grad, fx)
            return z, fz

        if self.bounds is not None and self.bounds.is_violated(next_point):
            self.logger.debug("Line-search on box constraint.")
            grad = CArray(x - self.bounds.projection(next_point))
            z, fz = self._line_search.minimize(x, -grad, fx)
            return z, fz

        z, fz = self._line_search.minimize(x, -grad, fx)
        return z, fz

    def minimize(self, x_init, args=(), **kwargs):
        """
        Interface to minimizers implementing
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

        # initialize line search (and re-assign fun to it)
        self._init_line_search(eta=self.eta,
                               eta_min=self.eta_min,
                               eta_max=self.eta_max)

        # constr.radius = 0, exit
        if self.constr is not None and self.constr.radius == 0:
            # classify x0 and return
            x0 = self.constr.center
            if self.bounds is not None and self.bounds.is_violated(x0):
                import warnings
                warnings.warn(
                    "x0 " + str(x0) + " is outside of the given bounds.",
                    category=RuntimeWarning)
            self._x_seq = CArray.zeros(
                (1, x0.size), sparse=x0.issparse, dtype=x0.dtype)
            self._f_seq = CArray.zeros(1)
            self._x_seq[0, :] = x0
            self._f_seq[0] = self._fun.fun(x0, *args)
            self._x_opt = x0
            return x0

        x = x_init.deepcopy()  # TODO: IS DEEPCOPY REALLY NEEDED?

        # if x is outside of the feasible domain, project it
        if self.bounds is not None and self.bounds.is_violated(x):
            x = self.bounds.projection(x)

        if self.constr is not None and self.constr.is_violated(x):
            x = self.constr.projection(x)

        if (self.bounds is not None and self.bounds.is_violated(x)) or \
                (self.constr is not None and self.constr.is_violated(x)):
            raise ValueError(
                "x_init " + str(x) + " is outside of feasible domain.")

        # dtype depends on x and eta (the grid discretization)
        if np.issubdtype(x_init.dtype, np.floating):
            # x is float, res dtype should be float
            self._dtype = x_init.dtype
        else:  # x is int, res dtype depends on the grid discretization
            self._dtype = self._line_search.eta.dtype

        # initialize x_seq and f_seq
        self._x_seq = CArray.zeros((self.max_iter, x_init.size),
                                   sparse=x_init.issparse,
                                   dtype=self._dtype)
        self._f_seq = CArray.zeros(self.max_iter)

        # The first point is obviously the starting point,
        # and the constraint is not violated (false...)
        fx = self._fun.fun(x, *args)  # eval fun at x, for iteration 0
        self._x_seq[0, :] = x
        self._f_seq[0] = fx

        self.logger.debug('Iter.: ' + str(0) + ', f(x): ' + str(fx))

        for i in range(1, self.max_iter):

            # update point
            x, fx = self._xk(x, fx, *args)

            # Update history
            self._x_seq[i, :] = x
            self._f_seq[i] = fx
            self._x_opt = x

            self.logger.debug('Iter.: ' + str(i) +
                              ', f(x): ' + str(fx) +
                              ', norm(gr(x)): ' +
                              str(CArray(self._grad).norm()))

            diff = abs(self.f_seq[i].item() - self.f_seq[i - 1].item())

            if diff < self.eps:
                self.logger.debug(
                    "Flat region, exiting... ({:.4f} / {:.4f})".format(
                        self._f_seq[i].item(), self._f_seq[i - 1].item()))
                self._x_seq = self.x_seq[:i + 1, :]
                self._f_seq = self.f_seq[:i + 1]
                return x

        self.logger.warning('Maximum iterations reached. Exiting.')
        return x
