"""
.. module:: CSolverGradDesc
   :synopsis: This class optimize with the standard gradient descent approach

.. moduleauthor:: Battista Biggio <battista.biggio@diee.unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>
"""
from secml.array import CArray
from secml.adv.attacks.evasion.solvers import CSolver
from secml.core.constants import nan, inf


class CSolverGradDesc(CSolver):
    """
    This is an abstract class for optimizing:

        min  f(x)
        s.t. d(x,x0) <= dmax
             x_lb <= x <= x_ub

    f(x) is the objective function (either linear or nonlinear),
    d(x,x0) <= dmax is a distance constraint in feature space (l1 or l2),
    and x_lb <= x <= x_ub is a box constraint on x.

    The solution algorithm is based on the classic gradient descent algorithm.
    """

    class_type = 'gradient-descent'

    def __init__(self, fun,
                 constr=None,
                 bounds=None,
                 discrete=False,
                 eta=1e-3,
                 eps=1e-4,
                 max_iter=200):

        CSolver.__init__(self, fun=fun,
                         constr=constr, bounds=bounds,
                         discrete=discrete)

        if discrete is True:
            raise NotImplementedError(
                'Descent in discrete space not implemented!')

        self._eta = None  # gradient step size
        self._eps = None  # tolerance value for stop criterion
        self._max_iter = None  # maximum number of iterations

        # invoke setters
        self.eta = eta
        self.max_iter = max_iter
        self.eps = eps

        CSolverGradDesc.__clear(self)

    def __clear(self):
        pass

    ###########################################################################
    #                              SETTER-GETTER
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

    ###########################################################################
    #                           READ-WRITE ATTRIBUTES
    ###########################################################################

    # none (by now)

    ###########################################################################
    #                             PRIVATE METHODS
    ###########################################################################

    # none (by now)

    ###########################################################################
    #                             PUBLIC METHODS
    ###########################################################################

    def minimize(self, x_init):
        """
        Interface to minimizers implementing
            min fun(x)
            s.t. constraint

        Parameters:
        ------
        x:
            the initial input point
        eps:
            a small number to check convergence

        Returns:
        ------
        self.f_seq:
            CArray containing values of f during optimization
        self.x_seq:
            CArray containing values of x during optimization
        """

        self._fun.clear()  # reset fun and grad evaluation counts

        x = x_init.deepcopy()

        if self.constr is not None and self.constr.is_violated(x):
            x = self.constr.projection(x)

        if self.bounds is not None and self.bounds.is_violated(x):
            x = self.bounds.projection(x)

        self._x_seq = CArray.zeros((self._max_iter, x.size))
        self._f_seq = CArray.zeros(self._max_iter)

        for i in xrange(self._max_iter):

            self._x_seq[i, :] = x
            self._f_seq[i] = self._fun.fun(x)

            # if i > 0 and abs(self.f_seq[i - 1] - self.f_seq[i]) < self.eps:
            #     # if i > 0 and self.f_seq[i - 1] - self.f_seq[i] < self.eps:
            #     self.logger.debug("Flat region, exiting... {:}  {:}".format(
            #         self._f_seq[i],
            #         self._f_seq[i - 1]))
            #     self._x_seq = self.x_seq[:i, :]
            #     self._f_seq = CArray(self.f_seq[:i])
            #     self._x_opt = self._x_seq[-1, :]
            #     return self._x_opt

            if i > 6:
                if self.f_seq[-3:].mean() < self.f_seq[-6:-3].mean():
                    self.logger.debug("Decreasing function, exiting... {:}  {:}".format(
                        self.f_seq[-3:].mean(),
                        self.f_seq[-6:-3].mean()
                    ))
                    self._x_seq = self.x_seq[:i-3, :]
                    self._f_seq = self.f_seq[:i-3]
                    self._x_opt = self._x_seq[-4, :]
                    return self._x_opt

            grad = self._fun.gradient(x)

            # debugging information
            self.logger.debug('Iter.: ' + str(i) + ', x: ' + str(x) +
                              ', f(x): ' + str(self._f_seq[i]) + '|g(x)|_2: ' + str(grad.norm()))

            # make a step into the deepest descent direction
            x -= self.eta * grad

            # project x onto the feasible domain
            if self.constr is not None and self.constr.is_violated(x):
                x = self.constr.projection(x)
            if self.bounds is not None and self.bounds.is_violated(x):
                x = self.bounds.projection(x)

            # update fun/grad evaluations
            self._f_eval = self._fun.n_fun_eval
            self._grad_eval = self._fun.n_grad_eval

        # self.logger.warning('Maximum iterations reached. Exiting.')
        self._x_seq = self.x_seq[:self._max_iter - 1, :]
        self._f_seq = self.f_seq[:self._max_iter - 1]
        self._x_opt = self._x_seq[-1, :]
        return self._x_opt
