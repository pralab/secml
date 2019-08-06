"""
.. module:: COptimizerPGD
   :synopsis: Optimizer using Projected Gradient Descent

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from six.moves import range

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

        x = x_init.deepcopy()

        if self.constr is not None and self.constr.is_violated(x):
            x = self.constr.projection(x)

        if self.bounds is not None and self.bounds.is_violated(x):
            x = self.bounds.projection(x)

        self._x_seq = CArray.zeros((self._max_iter, x.size))
        self._f_seq = CArray.zeros(self._max_iter)

        for i in range(self._max_iter):

            self._x_seq[i, :] = x
            self._f_seq[i] = self._fun.fun(x, *args)

            if i > 0 and abs(self.f_seq[i - 1] - self.f_seq[i]) < self.eps:
                self.logger.debug("Flat region, exiting... {:}  {:}".format(
                    self._f_seq[i], self._f_seq[i - 1]))
                self._x_seq = self.x_seq[:i, :]
                self._f_seq = self.f_seq[:i]
                self._x_opt = self._x_seq[-1, :]
                return self._x_opt

            if i > 6 and self.f_seq[-3:].mean() < self.f_seq[-6:-3].mean():
                self.logger.debug(
                    "Decreasing function, exiting... {:}  {:}".format(
                        self.f_seq[-3:].mean(), self.f_seq[-6:-3].mean()))
                self._x_seq = self.x_seq[:i-3, :]
                self._f_seq = self.f_seq[:i-3]
                self._x_opt = self._x_seq[-4, :]
                return self._x_opt

            grad = self._fun.gradient(x, *args)

            # debugging information
            self.logger.debug(
                'Iter.: ' + str(i) + ', x: ' + str(x) + ', f(x): ' +
                str(self._f_seq[i]) + '|g(x)|_2: ' + str(grad.norm()))

            # make a step into the deepest descent direction
            x -= self.eta * grad

            # project x onto the feasible domain
            if self.constr is not None and self.constr.is_violated(x):
                x = self.constr.projection(x)
            if self.bounds is not None and self.bounds.is_violated(x):
                x = self.bounds.projection(x)

        # self.logger.warning('Maximum iterations reached. Exiting.')
        self._x_seq = self.x_seq[:self._max_iter - 1, :]
        self._f_seq = self.f_seq[:self._max_iter - 1]
        self._x_opt = self._x_seq[-1, :]

        return self._x_opt
