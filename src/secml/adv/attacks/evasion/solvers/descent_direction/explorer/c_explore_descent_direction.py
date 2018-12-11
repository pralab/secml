"""
.. module:: CExploreDescentDirection
   :synopsis: This class explores a descent direction. Differently from
   standard line searches, it explores a subset of n_dimensions at a time.
   In this sense, it is an extension of the classical line-search approach.

.. moduleauthor:: Battista Biggio <battista.biggio@diee.unica.it>
"""
from secml.core import CCreator
from secml.optimization.line_search import CLineSearch
from secml.array import CArray


class CExploreDescentDirection(CCreator):
    """

    Attributes
    ----------
    class_type : 'descent-direction'

    """
    __class_type = 'descent-direction'

    def __init__(self, fun, constr=None, bounds=None,
                 n_dimensions=0, line_search='bisect',
                 eta=1e-3, eta_min=None, eta_max=None, max_iter=50):

        """Constructor for the class."""
        self._n_dimensions = n_dimensions

        self._line_search = CLineSearch.create(
            line_search,
            fun=fun,
            constr=constr,
            bounds=bounds,
            eta=eta,
            eta_min=eta_min,
            eta_max=eta_max,
            max_iter=max_iter,
            stop_criterion='armijo-goldstein')

        # TODO fix verbose - this is hardcoded
        self._line_search.verbose = 2

        CExploreDescentDirection.__clear(self)
        return

    @property
    def n_dimensions(self):
        return self._n_dimensions

    @n_dimensions.setter
    def n_dimensions(self, val):
        self._n_dimensions = int(val)

    @property
    def eta(self):
        return self._line_search.eta

    @property
    def eta_max(self):
        return self._line_search.eta_max

    @eta_max.setter
    def eta_max(self, value):
        self._line_search.eta_max = value

    @property
    def fun(self):
        return self._line_search.fun

    @property
    def constr(self):
        return self._line_search.constr

    @property
    def bounds(self):
        return self._line_search.bounds

    def __clear(self):
        """
        Re-initializes this class instance,
        restoring its status right after
        the __init__ call.
        """

        # descent direction (passed from outside)
        self._descent_direction = None

        # this is useful to check if exploration is finished
        self._explored = False

        # Indicies of features that would sort the gradient
        # in descending order of their absolute value
        # (otherwise, for random direction, randomly set to +1 or -1)
        self._idx_top_feat = None

        # This index vector indexes _idx_top_feat,
        # starting from 0 to n_neighbors-1.
        # Features which do not generate useful candidates
        # (e.g., constraint violations) are substitued with
        # next candidate features, until all features are explored
        self._idx_current = None

        # index of next candidate feature
        self._idx_next = None

        # number of features
        self._n_feat = None

        # clear line search
        self._line_search.clear()

    def finished_exploration(self):
        """
        This function returns True
        if the exploration of the feature
        (dimension) subsets is terminated.
        """
        if self._explored is True:
            return True
        return False

    def set_descent_direction(self, x):
        """
        This function sets the descent direction to either the gradient of fun
        (if fun is differentiable) or to a random direction (otherwise), and
        reset exploration of its dimension subsets.
        """
        self._n_feat = x.size

        if self._n_dimensions <= 0 or self._n_dimensions > self._n_feat:
            self._n_dimensions = self._n_feat

        self.reset_exploration()

        # TODO: remove sorting from inside set_gradient, do it after filtering
        if self.fun.has_gradient():
            self._set_gradient_descent_direction(x)
        else:
            self._set_random_descent_direction(x)

        # remove features that, if modified, violate the box constraint
        x_feas = self._filter_descent_direction(x)

        # if we are optimizing all features at once,
        # there's no need of sorting to find the best ones
        if self._n_dimensions < self._n_feat:
            # TODO: see argpartition (avoid sorting the whole array!)
            self.logger.info("Warning. Sorting full array.")
            self._idx_top_feat = (-abs(self._descent_direction)).argsort()
        else:  # TODO: this is not required, we should avoid it.
            self._idx_top_feat = CArray.randsample(x.size, x.size)

        # TODO: this may be not required now.
        self._ff_idx_top_feat = self._idx_top_feat[
            x_feas[self._idx_top_feat] == 1]

        self._ff_n_feat = self._ff_idx_top_feat.size

    def _filter_descent_direction(self, x):
        """Exclude from descent direction those features which,
        if modified according to the given descent direction,
        would violate the box constraint.
        """
        if self._descent_direction is None:
            return

        x_lb = 1 - (
            (self._descent_direction > 0)).logical_and(
            x.round(12) == self.bounds.lb.round(12)).astype(int)

        x_ub = 1 - (
            (self._descent_direction < 0)).logical_and(
            x.round(12) == self.bounds.ub.round(12)).astype(int)

        # feasible feature manipulations that do not violate box
        x_feas = x_lb * x_ub

        # reset gradient for unfeasible features
        self._descent_direction[x_feas == 0] = 0

        return x_feas

    def _set_gradient_descent_direction(self, x):
        """Sets the descent direction to the gradient of fun"""
        self._descent_direction = self.fun.gradient(x)

    def _set_random_descent_direction(self, x):
        """
        Generates a random descent direction consisting of
        +1/-1 with equal probability
        """
        idx = CArray.randsample(x.size, int(0.5 * x.size))
        self._descent_direction = CArray.ones(x.size)
        self._descent_direction[idx] = -self._descent_direction[idx]
        self._idx_top_feat = CArray.randsample(x.size, x.size)

    def _current_descent_direction(self):
        """
        This function returns the descent direction
        to be explored, namely, the one obtained
        by projecting the given direction
        onto the current subset of dimensions.
        """
        idx = self._ff_idx_top_feat
        f = self._idx_current

        if idx.size == 0:  # no candidate features to be modified
            return CArray.zeros(
                shape=(1, self._n_feat),
                sparse=self._descent_direction.issparse,
                dtype=self._descent_direction.dtype).ravel()

        # descent direction (exploring n features at a time)
        if self._n_dimensions == self._n_feat:
            d = self._descent_direction
        else:
            d = CArray.zeros(shape=(1, self._n_feat),
                             sparse=self._descent_direction.issparse,
                             dtype=self._descent_direction.dtype).ravel()
            d[idx[f]] = self._descent_direction[idx[f]]

        d_norm = d.ravel().norm()
        if d_norm < 1e-21:
            return CArray.zeros(shape=(1, self._n_feat),
                                sparse=d.issparse,
                                dtype=d.dtype).ravel()
        return d / d_norm

    # questa e' la reset di update_current_subset
    # agganciare a warm_start...
    def reset_exploration(self):
        self._explored = False
        self._idx_next = self._n_dimensions
        self._idx_current = CArray.arange(0, self._idx_next, 1)
        self._line_search.clear()
        return

    def _update_current_subset(self):
        """
        Updates the current set of features to be explored.
        For instance, if n = 5, our algorithm starts exploring
        features 0, 1, 2, 3, 4 (indexing the most relevant features
        from the current descent direction).
        This function will update this set to
        features 5, 6, 7, 8 after optimization of the former ones.
        This happens until the objective is minimized,
        and until all features have been explored.
        """
        if self._idx_next >= self._ff_n_feat:
            self._explored = True  # exploration has finished
            return
        self._idx_next += self._idx_current.size

        self._idx_current = CArray.arange(0, self._idx_next, 1)

        self._idx_current = self._idx_current[
            self._idx_current < self._ff_n_feat]

        self._idx_next = min(self._idx_next, self._ff_n_feat)

    def explore_descent_direction(self, x, fx):
        """
        Generates the first point that minimizes fun(x - eta*d) by exploring
        the current descent direction n_dimensions at a time.
        Returns the same point x if no better point is found
        along the given direction.
        """
        x = x.ravel()
        score = fx

        while not self.finished_exploration() and self._ff_n_feat > 0:

            # descent direction (exploring n features at a time)
            d = self._current_descent_direction()

            if d.ravel().norm() < 1e-20:
                return x, score

            if self.constr is not None and \
                    self.constr.is_violated(x - self.eta * d):
                return x, score

            if self.bounds is not None and \
                    self.bounds.is_violated(x - self.eta * d):
                return x, score

            # line search executed in feature space
            v, fv = self._line_search.line_search(
                x=x, d=-d, grad=self._descent_direction, fx=score)

            # update subset and continue exploration
            self._update_current_subset()

            if fv < score:
                # return current point (better than x)
                return v, fv

        # no better point found
        return x, score
