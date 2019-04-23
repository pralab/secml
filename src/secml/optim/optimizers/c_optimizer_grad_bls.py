"""
.. module:: COptimizerGradBLS
   :synopsis: This class explores a descent direction using Bisect Line Search.
   Differently from standard line searches, it explores a subset of
   n_dimensions at a time. In this sense, it is an extension of the
   classical line-search approach.

.. moduleauthor:: Battista Biggio <battista.biggio@diee.unica.it>

"""
from six.moves import range
import numpy as np

# Used only by this class, will be removed in the future
from secml.optim.optimizers.explorer import _CExploreDescentDirection

from secml.array import CArray
from secml.optim.optimizers import COptimizer


class COptimizerGradBLS(COptimizer):
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
    problems, where x is integer valued. In this case, exploration works
    by manipulating one feature at a time.

    Attributes
    ----------
    class_type : 'gradient-bls'

    """
    __class_type = 'gradient-bls'

    def __init__(self, fun,
                 constr=None, bounds=None,
                 discrete=False,
                 eta=1e-3,
                 eta_min=None,
                 eta_max=None,
                 max_iter=1000,
                 eps=1e-4):

        COptimizer.__init__(self, fun=fun,
                            constr=constr, bounds=bounds,
                            discrete=discrete)

        # Read/write attributes
        self.eta = eta
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.max_iter = max_iter
        self.eps = eps

        # Internal attributes
        self._explorer = None

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

    def _initialize_explorer(
            self, line_search, eta, eta_min, eta_max, discrete):
        """Initialize explorer."""

        if discrete is True:
            if self.constr is not None and self.constr.class_type == 'l2':
                # TODO: CHECK SUPPORT OF DISCRETE + L2
                raise NotImplementedError(
                    "L2 constraint is not supported for discrete optimization")
            # Discrete optimization but no l2 constraint (not supported)
            n_dimensions = 1
        elif self.constr is not None and self.constr.class_type == 'l1':
            # Continue optimization and l1 constraint
            n_dimensions = 1
        elif self.constr is not None and self.constr.class_type == 'l2':
            # Continue optimization and l2 constraint
            n_dimensions = 0
        elif self.constr is None:
            # Continue optimization and no constraint
            n_dimensions = 0
        else:
            raise NotImplementedError(
                "the chosen combination of solver parameters is not supported")

        self._explorer = _CExploreDescentDirection(
            fun=self._fun,
            constr=self._constr,
            bounds=self._bounds,
            n_dimensions=n_dimensions,
            line_search=line_search,
            eta=eta, eta_min=eta_min, eta_max=eta_max,
            discrete=discrete)

        # TODO: fix this (decide whether propagate verbosity level or not)
        self._explorer.verbose = 0  # self.verbose
        self._explorer._line_search.verbose = 0  # self.verbose

    def _xk(self, x, fx):
        """Returns a new point after gradient descent."""

        # compute gradient
        self._explorer.set_descent_direction(x)
        grad = self._explorer._descent_direction

        norm = grad.norm()
        if norm < 1e-20:
            return x, fx  # return same point (and exit optimization)

        grad /= norm  # normalize gradient to unit norm

        # if constraint is active (also in discrete space)
        # run line search on the projection of the gradient onto the constraint
        if self._is_constr_violated(x, self._constr):
            self.logger.debug("Line-search on distance constraint.")
            return self._descent_direction_on_constr(x, fx, grad, self._constr)

        # if box is active (also in discrete space)
        # run line search on the projection of the gradient onto the box
        if self._is_constr_violated(x, self._bounds):
            self.logger.debug("Line-search on box constraint.")
            return self._descent_direction_on_constr(x, fx, grad, self._bounds)

        # ... otherwise run line search along the gradient direction
        z, fz = self._explorer.explore_descent_direction(x, fx)

        return z, fz

    def _is_constr_violated(self, z, constr):
        # in discrete spaces, or for large eta values
        # the point may be not sufficiently close
        # to activate the constraint. For this reason, we check whether
        # an update on the current direction would violate the constraint.
        # If this happens, then we consider the constraint active.

        # no constraint, no violation
        if constr is None:
            return False

        d = self._explorer._current_descent_direction()
        next_point = z - self._explorer.eta * d

        if constr.is_violated(next_point):
            return True

        return False

    def _descent_direction_on_constr(self, x, fx, grad, constr):
        """
        Finds a descent direction parallel to the active constraint surface
        """

        d = self._explorer._current_descent_direction()
        next_point = x - d * self._explorer.eta
        x_constr = constr.projection(next_point)

        # assuming gradient on x_constr to be equal to that in x
        if x_constr.issparse:
            u = CArray(x_constr.nnz_data).todense()
            if self._explorer.eta.size == 1:
                u -= self._explorer.eta * grad[x_constr.nnz_indices]
            else:
                u -= self._explorer.eta[x_constr.nnz_indices] * \
                     grad[x_constr.nnz_indices]
            v = x_constr.deepcopy()
            v[v.nnz_indices] = u
        else:
            v = x_constr - self._explorer.eta * grad

        v = constr.projection(v)
        d = CArray(v - x_constr)

        if d.ravel().norm() < 1e-20:
            return x, fx

        z, fz = self._explorer._line_search.line_search(x, d, fx=fx)
        return z, fz

    def minimize(self, x):
        """
        Interface to minimizers implementing
            min fun(x)
            s.t. constraint

        Parameters
        ----------
        x : CArray
            The initial input point.

        Returns
        -------
        f_seq : CArray
            Array containing values of f during optimization.
        x_seq : CArray
            Array containing values of x during optimization.

        """
        # reset fun and grad eval counts for both fun and f (by default fun==f)
        self._f.reset_eval()
        self._fun.reset_eval()

        # initialize explorer
        self._initialize_explorer(line_search='bisect',
                                  eta=self.eta,
                                  eta_min=self.eta_min,
                                  eta_max=self.eta_max,
                                  discrete=self.discrete)

        # constr.radius = 0, exit
        if self.constr is not None and self.constr.radius == 0:
            # classify x0 and return
            x0 = self.constr.center
            self._x_seq = CArray.zeros(
                (1, x0.size), sparse=x0.issparse, dtype=x0.dtype)
            self._f_seq = CArray.zeros(1)
            self._x_seq[0, :] = x0
            self._f_seq[0] = self._fun.fun(x0)
            self._x_opt = x0
            return

        # eval fun at x
        fx = self._fun.fun(x)

        # TODO: fix this part / separa per BOX e CONSTR!
        # line search in feature space towards the benign data.
        # if x is outside of the feasible domain, we run a line search to
        # identify the closest point to x in feasible domain (starting from x0)
        if self.bounds is not None and self.bounds.is_violated(x):
            x = self.bounds.projection(x)

        if self.constr is not None and self.constr.is_violated(x):
            d = x - self._constr.center  # direction in feature space
            if self.discrete:
                d = d.sign()
            self._explorer.eta_max = 2.0 * self.constr.constraint(x)
            x, fx = self._explorer._line_search.line_search(
                self._constr.center, d)
            self._explorer.eta_max = self.eta_max

        if (self.bounds is not None and self.bounds.is_violated(x)) or \
                (self.constr is not None and self.constr.is_violated(x)):
            raise ValueError("x " + str(x) + " is outside of feasible domain.")

        self._x_seq = CArray.zeros(
            (self.max_iter, x.size), sparse=x.issparse, dtype=x.dtype)
        self._f_seq = CArray.zeros(self.max_iter)

        # The first point is obviously the starting point,
        # and the constraint is not violated (false...)
        self._x_seq[0, :] = x
        self._f_seq[0] = fx

        # debugging information
        # self.logger.debug('Iter.: ' + str(0) + ', x: ' + str(x) +
        #                   ', f(x): ' + str(fx))

        self.logger.debug('Point optim iter.: ' + str(0) +
                          ', f(x): ' + str(fx))

        for i in range(1, self.max_iter):

            # update point
            x, fx = self._xk(x, fx=fx)

            if np.issubdtype(x.dtype, np.floating):
                # The current point is float,
                # so we need to update the type of x_sex
                self._x_seq = self._x_seq.astype(float)

            # Update history
            self._x_seq[i, :] = x
            self._f_seq[i] = fx
            self._x_opt = x

            self.logger.debug('Iter.: ' + str(i) +
                              ', f(x): ' + str(fx) +
                              ', norm(gr(x)): ' +
                              str(self._explorer._descent_direction.norm()))

            diff = abs(self.f_seq[i].item() - self.f_seq[i - 1].item())

            self.logger.debug('delta_f: {:}'.format(diff))

            if diff < self.eps:
                self.logger.debug("Flat region, exiting... {:}  {:}".format(
                    self._f_seq[i].item(),
                    self._f_seq[i - 1].item()))
                self._x_seq = self.x_seq[:i + 1, :]
                self._f_seq = self.f_seq[:i + 1]
                return x

        self.logger.warning('Maximum iterations reached. Exiting.')

        return x
