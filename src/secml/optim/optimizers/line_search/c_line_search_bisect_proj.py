"""
.. module:: CLineSearchBisectProj
   :synopsis: Binary line search including projections.

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>

"""
import numpy as np

from secml.array import CArray
from secml.optim.optimizers.line_search import CLineSearchBisect


class CLineSearchBisectProj(CLineSearchBisect):
    """Binary line search including projections.

    Parameters
    ----------
    fun : CFunction
        The function to use for the optimization.
    constr : CConstraintL1 or CConstraintL2 or None, optional
        A distance constraint. Default None.
    bounds : CConstraintBox or None, optional
        A box constraint. Default None.
    eta : scalar, optional
        Minimum resolution of the line-search grid. Default 1e-4.
    eta_min : scalar or None, optional
        Initial step of the line search. Gets multiplied or divided by 2
        at each step until convergence. If None, will be set equal to eta.
        Default 0.1.
    eta_max : scalar or None, optional
        Maximum step of the line search. Default None.
    max_iter : int, optional
        Maximum number of iterations of the line search. Default 20.

    Attributes
    ----------
    class_type : 'bisect-proj'

    """
    __class_type = 'bisect-proj'

    def __init__(self, fun, constr=None, bounds=None,
                 eta=1e-4, eta_min=0.1, eta_max=None,
                 max_iter=20):

        CLineSearchBisect.__init__(
            self, fun=fun, constr=constr, bounds=bounds,
            eta=eta, eta_min=eta_min, eta_max=eta_max, max_iter=max_iter)

        self._best_score = None
        self._best_eta = None
        self._dtype = None

    def _update_z(self, x, eta, d, projection=False):
        """Update z and its cached score fz."""
        z = CArray(x + eta * d, dtype=self._dtype, tosparse=x.issparse)
        if projection:
            z = self.bounds.projection(z) if self.bounds is not None else z
            z = self.constr.projection(z) if self.constr is not None else z
        self._fz = self.fun.fun(z)
        if self._fz < self._best_score:
            self._best_score = self._fz
            self._best_eta = eta
        return z

    def _is_feasible(self, x):
        """Checks if x is within the feasible domain."""
        constr_violation = False if self.constr is None else \
            self.constr.is_violated(x)
        bounds_violation = False if self.bounds is None else \
            self.bounds.is_violated(x)

        if constr_violation or bounds_violation:
            return False

        return True

    def _select_best_point(self, x, d, idx_min, idx_max, **kwargs):
        """Returns best point among x and the two points found by the search.
        In practice, if f(x + eta*d) increases on d, we return x."""

        v = CArray(x + d * self._best_eta,
                   dtype=self._dtype, tosparse=x.issparse)
        if self.bounds is not None:
            v = self.bounds.projection(v) if self.bounds is not None else v
        if self.constr is not None:
            v = self.constr.projection(v) if self.bounds is not None else v
        if self._is_feasible(v):
            return v, self._best_score

        x1 = CArray(x + d * self.eta * idx_min,
                    dtype=self._dtype, tosparse=x.issparse)
        if self.bounds is not None:
            x1 = self.bounds.projection(x1) if self.bounds is not None else x1
        if self.constr is not None:
            x1 = self.constr.projection(x1) if self.constr is not None else x1

        x2 = CArray(x + d * self.eta * idx_max,
                    dtype=self._dtype, tosparse=x.issparse)
        if self.bounds is not None:
            x2 = self.bounds.projection(x2) if self.bounds is not None else x2
        if self.constr is not None:
            x2 = self.constr.projection(x2) if self.constr is not None else x2

        self.logger.debug("Select best point between: f[a], f[b]: [" +
                          str(self._fun_idx_min) + "," +
                          str(self._fun_idx_max) + "]")
        f0 = self._fx

        if not self._is_feasible(x1) and \
                not self._is_feasible(x2):
            if self._best_score < f0:
                self.logger.debug("x1 and x2 are not feasible."
                                  "Returning the best cached value.")
                return v, self._best_score
            else:
                self.logger.debug("x1 and x2 are not feasible. Returning x.")
                return x, f0

        # uses cached values (if available) to save computations
        f1 = self._fun_idx_min if self._fun_idx_min is not None else \
            self.fun.fun(x1, **kwargs)

        if not self._is_feasible(x2):
            if f1 < f0 and f1 < self._best_score:
                self.logger.debug("x2 not feasible. Returning x1."
                                  " f(x): " + str(f0) +
                                  ", f(x1): " + str(f1))
                return x1, f1
            if self._best_score < f1:
                self.logger.debug("x2 not feasible. Returning the best cached"
                                  " value.")
                return v, self._best_score
            self.logger.debug("x2 not feasible. Returning x."
                              " f(x): " + str(f0) +
                              ", f(x1): " + str(f1))
            return x, f0

        # uses cached values (if available) to save computations
        f2 = self._fun_idx_max if self._fun_idx_max is not None else \
            self.fun.fun(x2, **kwargs)

        if not self._is_feasible(x1):
            if f2 < f0 and f2 < self._best_score:
                self.logger.debug("x1 not feasible. Returning x2.")
                return x2, f2
            if self._best_score < f2:
                self.logger.debug("x1 not feasible. Returning the best cached "
                                  "value.")
                return v, self._best_score
            self.logger.debug("x1 not feasible. Returning x.")
            return x, f0

        # else return best point among x1, x2 and x
        self.logger.debug("f0: {:}, f1: {:}, f2: {:}, best: {:}".format(
            f0, f1, f2, self._best_score))

        if f2 <= f0 and f2 <= f1 and f2 <= self._best_score:
            self.logger.debug("Returning x2.")
            return x2, f2

        if f1 <= f0 and f1 <= f2 and f1 <= self._best_score:
            self.logger.debug("Returning x1.")
            return x1, f1

        if self._best_score <= f0 and self._best_score <= f1 and \
                self._best_score <= f2:
            self.logger.debug("Returning the best cached value.")
            return v, self._best_score

        self.logger.debug("Returning x.")
        self.logger.debug("f0: {:}".format(f0))
        return x, f0

    def _is_decreasing(self, x, d, **kwargs):
        """
        Returns True if function at `x + eps*d` is decreasing,
        or False if it is increasing or out of feasible domain.
        """
        # IMPORTANT: requires self._fz to be set to fun.fun(z)
        # This is done to save function evaluations

        if not self._is_feasible(x):
            # point is outside of feasible domain
            return False

        # this could be in the order of 1e-10 or 1e-12, if eta is very small
        z1 = x + 0.1 * self.eta * d
        z1 = self.bounds.projection(z1) if self.bounds is not None else z1
        z1 = self.constr.projection(z1) if self.constr is not None else z1
        delta = self.fun.fun(z1, **kwargs) - self._fz

        if delta <= 0:
            # feasible point, decreasing / stationary score
            return True

        # feasible point, increasing score
        return False

    def _compute_eta_max(self, x, d, **kwargs):

        # double eta each time until function increases or goes out of bounds
        eta = self.eta if self.eta_min is None else self.eta_min

        # eta_min may be too large, going out of bounds,
        # or jumping out of the local minimum
        # it this happens, we reduce it,
        # ensuring a feasible point or a minimal step (multiple of self.eta)
        # this helps getting closer to the violated constraint
        t = CArray(eta / self.eta).round()

        # FIXME: MANY UNUSED VARIABLES IN THE FOLLOWING

        # update z and fz
        z = self._update_z(x, eta, d, projection=True)

        # divide eta by 2 if x+eta*d goes out of bounds or fz decreases
        # update (if required) z and fz
        while eta > self.eta and self._fz > self._fx:
            t = CArray(t / 2).round()
            eta = t * self.eta

            # store fz (for current point)
            z = self._update_z(x, eta, d, projection=True)

            self.logger.debug(
                "[_compute_eta_min] eta: " + str(eta.item()) +
                ", f(z): " + str(self._fz))

        # exponential line search starts here
        while self._n_iter < 10:
            # cache f_min
            f_min_old = CArray(self._fun_idx_min)
            self._fun_idx_min = self._fz

            eta *= 2

            # update z and fz
            z = self._update_z(x, eta, d, projection=True)

            # cache f_max
            f_max_old = CArray(self._fun_idx_max)
            self._fun_idx_max = self._fz

            self.logger.debug(
                "[_compute_eta_max] eta: " + str(eta.item()) +
                ", f(z0): " + str(self._fun_idx_min) +
                ", f(z1): " + str(self._fun_idx_max))

            self._n_iter += 1

        self.logger.debug('Maximum iterations reached. Exiting.')
        return eta

    def minimize(self, x, d, fx=None, tol=1e-4, **kwargs):
        """Exponential line search (on discrete grid).

        The function `fun( x + a*eta*d )` with `a = {0, 1, 2, ... }`
        is minimized along the descent direction d.

        If `fun(x) >= 0` -> step_min = step
        else step_max = step

        If eta_max is not None, it runs a bisect line search in
        `[x + eta_min*d, x + eta_max*d];
        otherwise, it runs an exponential line search in
        `[x + eta*d, ..., x + eta_min*d, ...]`

        Parameters
        ----------
        x : CArray
            The input point.
        d : CArray
            The descent direction along which `fun(x)` is minimized.
        fx : int or float or None, optional
            The current value of `fun(x)` (if available).
        tol : float, optional
            Tolerance for convergence to the local minimum.
        kwargs : dict
            Additional parameters required to evaluate `fun(x, **kwargs)`.

        Returns
        -------
        x' : CArray
            Point `x' = x + eta * d` that approximately
            solves `min f(x + eta*d)`.
        fx': int or float or None, optional
            The value `f(x')`.

        """
        d = CArray(d).ravel()

        # dtype depends on x and eta (the grid discretization)
        if np.issubdtype(x.dtype, np.floating):
            # if x is float res dtype should be float
            self._dtype = x.dtype
        else:  # x is int, so the res dtype depends on the grid discretization
            self._dtype = self.eta.dtype

        self._n_iter = 0

        self._fx = self.fun.fun(x) if fx is None else fx
        self._fz = self._fx

        self._best_score = self._fx
        self._best_eta = 0.0

        self.logger.info(
            "line search: " +
            ", f(x): " + str(self._fx))

        # reset cached values
        self._fun_idx_min = None
        self._fun_idx_max = None

        # exponential search
        if self.eta_max is None:
            self.logger.debug("Exponential search ")
            eta_max = self._compute_eta_max(x, d, **kwargs)
            idx_max = (eta_max / self.eta).ceil().astype(int)
            idx_min = (idx_max / 2).astype(int)
            # this only searches within [eta, 2*eta]
            # the values fun_idx_min and fun_idx_max are already cached
        else:
            self.logger.debug("Binary search ")
            idx_max = (self.eta_max / self.eta).ceil().astype(int)
            idx_min = 0
            self._fun_idx_min = self._fx
            self._fun_idx_max = None  # this has not been cached

        self.logger.debug("Running line search in: f[a], f[b]: [" +
                          str(self._fun_idx_min) + "," +
                          str(self._fun_idx_max) + "]")

        return self._select_best_point(x, d, idx_min, idx_max, **kwargs)
