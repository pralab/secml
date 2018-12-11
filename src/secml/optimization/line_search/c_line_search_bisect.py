"""
.. module:: CLineSearchBisect
   :synopsis: Binary line search

.. moduleauthor:: Battista Biggio <battista.biggio@diee.unica.it>
"""

from secml.optimization.line_search.c_line_search import CLineSearch
from secml.array import CArray


class CLineSearchBisect(CLineSearch):
    """Implements binary line search.

    Attributes
    ----------
    class_type : 'bisect'

    """
    __class_type = 'bisect'

    def __init__(self, fun, constr=None, bounds=None,
                 eta=1e-4, eta_min=0.1, eta_max=None,
                 max_iter=20, stop_criterion='armijo-goldstein'):

        super(CLineSearchBisect, self).__init__(fun, constr, bounds,
                                                eta=eta,
                                                max_iter=max_iter)

        self._stop_criterion = None
        self.stop_criterion = stop_criterion

        self.eta_max = eta_max
        self.eta_min = eta_min

        # internal parameters
        self._n_iter = 0
        self._fx = None  # value of fun at x (initial point)
        self._fz = None  # value of fun at current z during line search

    @property
    def stop_criterion(self):
        return self._stop_criterion

    @stop_criterion.setter
    def stop_criterion(self, value):
        if value is None or value == 'armijo-goldstein':
            self._stop_criterion = value
        else:
            raise ValueError("Unknown stop criterion.")

    @property
    def eta_max(self):
        return self._eta_max

    @eta_max.setter
    def eta_max(self, value):
        """
        Sets eta_max to value (multiple of eta)

        Parameters
        ----------
        value: CArray or None
        """
        if value is None:
            self._eta_max = None
            return
        # set eta_max >= t*eta, t >= 1 (integer)
        t = max(CArray(value / self.eta).round(), 1)
        self._eta_max = self.eta * t

    @property
    def eta_min(self):
        return self._eta_min

    @eta_min.setter
    def eta_min(self, value):
        """
        Sets eta_min to value (multiple of eta)

        Parameters
        ----------
        value: CArray or None
        """
        if value is None:
            self._eta_min = None
            return
        # set eta_min >= t*eta, t >= 1 (integer)
        t = max(CArray(value / self.eta).round(), 1)
        self._eta_min = self.eta * t

    @property
    def n_iter(self):
        return self._n_iter

    def _is_feasible(self, x):

        constr_violation = False if self.constr is None else \
            self.constr.is_violated(x)
        bounds_violation = False if self.bounds is None else \
            self.bounds.is_violated(x)

        if constr_violation or bounds_violation:
            return False

        return True

    def _select_best_point(self, x, d, idx_min, idx_max, **kwargs):
        """
        Returns best point among x, and the two points found
        by the line search.
        In practice, if f(x + eta*d) increases on d, we return x.
        """
        x1 = x if idx_min == 0 else CArray(x + d * self.eta * idx_min,
                                           dtype=x.dtype, tosparse=x.issparse)
        x2 = x if idx_max == 0 else CArray(x + d * self.eta * idx_max,
                                           dtype=x.dtype, tosparse=x.issparse)

        f0 = self._fx

        if not self._is_feasible(x1) and \
                not self._is_feasible(x2):
            self.logger.debug("x1 and x2 are not feasible. Returning x.")
            return x, f0

        # FIXME: THIS fun_idx_max thing is not working
        # f1 = self._fun_idx_min if self._fun_idx_min is not None else \
        #     self.fun.fun(x1, **kwargs)
        f1 = self.fun.fun(x1, **kwargs)

        if not self._is_feasible(x2):
            if f1 < f0:
                self.logger.debug("x2 not feasible. Returning x1."
                                  " f(x): " + str(f0) +
                                  ", f(x1): " + str(f1))
                return x1, f1
            self.logger.debug("x2 not feasible. Returning x."
                              " f(x): " + str(f0) +
                              ", f(x1): " + str(f1))
            return x, f0

        # FIXME: THIS fun_idx_max thing is not working
        # f2 = self._fun_idx_max if self._fun_idx_max is not None else \
        #     self.fun.fun(x2, **kwargs)
        f2 = self.fun.fun(x2, **kwargs)

        if not self._is_feasible(x1):
            if f2 < f0:
                self.logger.debug("x1 not feasible. Returning x2.")
                return x2, f2
            self.logger.debug("x1 not feasible. Returning x.")
            return x, f0

        # else return best point among x1, x2 and x
        if f2 <= f0 and f2 < f1:
            self.logger.debug("Returning x2.")
            # print f0, f1, f2
            return x2, f2

        if f1 <= f0 and f1 < f2:
            self.logger.debug("Returning x1.")
            # print f0, f1, f2
            return x1, f1

        self.logger.debug("Returning x.")
        # print f0, f1, f2
        return x, f0

    def is_decreasing(self, x, d, **kwargs):
        """
        Returns True if function at x + eps*d is decreasing
        Returns False if function at x + eps*d is increasing
        (or out of feasible domain)

        In practice, this function checks the slope of f at x
        """

        # IMPORTANT: requires self._fz to be set to fun.fun(z)
        # This is done to save function evaluations

        if not self._is_feasible(x):
            # point is outside of feasible domain
            return False

        # this could be in the order of 1e-10 or 1e-12, if eta is very small
        delta = self.fun.fun(x + 0.1 * self.eta * d, **kwargs) - self._fz

        if delta <= 0:
            # feasible point, decreasing / stationary score
            return True

        # feasible point, increasing score
        return False

    def line_search(self, x, d, grad=None, fx=None, **kwargs):
        """
        Bisect line search (on discrete grid)

        f ( x + a*d ) = 0
        a = {0, a, 2a, ... eta}

        fun is checked against zero.
        If f(x) >= 0, step_min=step
        else step_max = step

        if eta_max is not None, it runs BINARY SEARCH in [x, x+eta_max*d].
        if eta_max is None, it runs EXPONENTIAL SEARCH starting from x.

        """

        d = CArray(d, tosparse=d.issparse).ravel()

        self._n_iter = 0

        # func eval
        self._fx = self.fun.fun(x) if fx is None else fx
        self._fun_idx_min = None
        self._fun_idx_max = None

        if self.eta_max is None:
            eta_max = self._compute_eta_max(x, d, **kwargs)
            idx_max = (eta_max / self.eta).ceil()
            idx_min = (idx_max / 2).astype(int)
        else:
            idx_max = (self.eta_max / self.eta).ceil()
            idx_min = 0
            self._fun_idx_min = self._fx

        self.logger.debug("Running binary line search in: [" +
                          str(self.eta * idx_min) + "," +
                          str(self.eta * idx_max) + "]")
        self.logger.debug("f[a], f[b]: [" +
                          str(self._fun_idx_min) + "," +
                          str(self._fun_idx_max) + "]")

        while self._n_iter < self.max_iter:

            if idx_min == 0:
                if (idx_max <= 1).any():
                    # local minimum found
                    return self._select_best_point(
                        x, d, idx_min, idx_max, **kwargs)
            else:
                if (idx_max - idx_min <= 1).any():
                    # local minimum found
                    return self._select_best_point(
                        x, d, idx_min, idx_max, **kwargs)

            if idx_min == 0:
                # TODO: this avoids problems when idx_min = 0
                # fix when CArray / sparse data handling is fixed.
                idx = (0.5 * idx_max).astype(int)
            else:
                idx = (0.5 * (idx_min + idx_max)).astype(int)

            z = x + self.eta * d * idx
            self._fz = self.fun.fun(z)

            self.logger.debug("eta: " + str(self.eta * idx) + ", z: " +
                              str(z[z != 0]) + ", f(z): " + str(self._fz))

            self._n_iter += 1

            if self.is_decreasing(z, d, **kwargs):
                idx_min = idx
                self._fun_idx_min = self._fz
            else:
                idx_max = idx
                self._fun_idx_max = self._fz

            # armijo condition (loose minimization)
            if self._is_feasible(z) and \
                    self._stop_criterion == 'armijo-goldstein' and \
                    self._armijo_goldstein(self._fx, self._fz,
                                           self.eta * idx, d, grad):
                self.logger.debug("Armijo-Goldstein condition met.")
                return self._select_best_point(
                    x, d, idx_min, idx_max, **kwargs)

        self.logger.debug('Maximum iterations reached. Exiting.')
        return self._select_best_point(x, d, idx_min, idx_max, **kwargs)

    def _compute_eta_max(self, x, d, **kwargs):

        # double eta each time until function increases or goes out of bounds
        eta = self.eta if self.eta_min is None else self.eta_min

        # eta_min may be too large, going out of bounds,
        # or jumping out of the local minimum
        # it this happens, we reduce it,
        # ensuring a feasible point or a minimal step (multiple of self.eta)
        # this helps getting closer to the violated constraint
        t = CArray(eta / self.eta).round()

        self._fz = self.fun.fun(x + eta * d)
        # print "fz, fx, is_feas(z): ", self._fz, \
        # self._fx, self._is_feasible(x + eta * d)
        # print eta > self.eta
        # print self._fz > self._fx
        while eta > self.eta and \
                (not self._is_feasible(x + eta * d) or self._fz > self._fx):
            t = CArray(t / 2).round()
            eta = t * self.eta
            # print "red. eta_min: ", eta, t, self._is_feasible(x + eta * d)

            # store value of fun before 2*eta (not necessarily fx!)
            self._fz = self.fun.fun(x + eta * d)

        while self._n_iter < self.max_iter:

            # update function values at idx_min before 2*eta
            self._fun_idx_min = self._fz

            eta *= 2
            z = x + eta * d

            # update function values at idx_max and current z
            self._fz = self.fun.fun(z)
            self._fun_idx_max = self._fz

            self.logger.debug("eta: " + str(eta) + ", z: " +
                              str(z[z != 0]) + ", f(z): " + str(self._fz))

            self._n_iter += 1

            # function started increasing or end of bounds
            if not self.is_decreasing(z, d, **kwargs):
                return eta

        self.logger.debug('Maximum iterations reached. Exiting.')
        return eta

    @staticmethod
    def _armijo_goldstein(fx, fz, alpha, d, grad=None, c1=1e-3):
        """
        Evaluates Armijo-Goldstein bound conditions on step size alpha.
            (1) f(z) < f(x) + c1 * alpha * grad.dot(d)
            (2) f(x) + (1-c1) * alpha * grad.dot(d) < f(z)
            where z = x + alpha*p
        """

        d_norm = CArray(d).norm()  # ensure d has unary norm

        if d_norm <= 1e-12:
            return True

        d /= d_norm

        if grad is None:
            grad = d

        # print fx, fz, alpha
        # print "armijo: ", c1 * alpha * grad.ravel().dot(d.ravel())
        # print "goldstein: ", (1 - c1) * alpha * grad.ravel().dot(d.ravel())

        # this is the Armijo condition
        cond1 = fz - fx <= c1 * alpha * grad.ravel().dot(d.ravel())

        # this is the Goldstein condition
        cond2 = (1 - c1) * alpha * grad.ravel().dot(d.ravel()) <= fz - fx

        return cond1 and cond2
