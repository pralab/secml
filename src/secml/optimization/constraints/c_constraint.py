'''

This module implements the generic constraint class
Implements equality/inequality constraints in the canonic form
    c(x) <= 0

@author: Battista Biggio
'''

from abc import ABCMeta, abstractmethod
from secml.core import CCreator
from secml.array import CArray


class CConstraint(CCreator):
    """Abstract class that defines basic methods for constraints."""
    __metaclass__ = ABCMeta
    __super__ = 'CConstraint'

    @property
    def constraint_type(self):
        """Defines constraint type."""
        raise NotImplementedError()

    @abstractmethod
    def _constraint(self, x):
        """Returns the left-hand-value c(x) in the constraint, to be compared against 0
            x: a single sample
        """
        raise NotImplementedError()

    # This is not abstract as some constraints may not be differentiable
    def _gradient(self, x):
        """Returns the gradient of c(x) at x
            x: a single sample
        """
        raise NotImplementedError()

    @abstractmethod
    def _projection(self, x):
        """Project x onto feasible domain / within the given constraint.
            x: a single sample

            No need to check whether constraint is violated.
            It's already done by projection(x)
        """
        raise NotImplementedError()

    def _is_active(self, x, tol=1e-4):
        """
        Returns true or false depending on whether
        the constraint is active (c(x)=0) or not.
            By default we assume constraints of the form c(x) <= 0
            x: a single sample
        Tolerance to compare c(x) against 0 is set to 1e-4 by default.
        """
        if abs(self._constraint(x)) <= tol:
            return True
        return False

    def is_active(self, X, tol=1e-4):
        """
        Works either on a array of len(data.shape)==2,
        each row representing a data sample,
        or on a single sample (len(data.shape)==1).
        In the former case, it returns a vector, otherwise a scalar.
        """

        if len(X.shape) == 1:
            return self._is_active(X.ravel(), tol)
        is_active = CArray.ones(X.shape[0])
        for i in xrange(0, X.shape[0]):
            is_active[i] = self._is_active(X[i, :].ravel(), tol)

        return is_active

    def is_violated(self, X):
        """
        Works either on a array of len(data.shape)==2,
        each row representing a data sample,
        or on a single sample (len(data.shape)==1).
        In the former case, it returns a vector, otherwise a scalar.
        """
        if len(X.shape) == 1 or X.shape[0] == 1:  # TODO check if 1d vector
            return self._is_violated(X.ravel())
        is_violated = CArray.ones(X.shape[0])
        for i in xrange(X.shape[0]):
            is_violated[i] = self._is_violated(X[i, :].ravel())  # TODO todense

        return is_violated

    def _is_violated(self, x, ndigits=4):
        """Returns true or false depending on whether
        the constraint is violated or not.
            By default we assume constraints of the form c(x) <= 0
            x: a single sample
        """
        if self._constraint(x).round(ndigits) > 0:
            return True
        return False

    def constraint(self, X):
        """
        Works either on a array of len(data.shape)==2,
        each row representing a data sample,
        or on a single sample (len(data.shape)==1).
        In the former case, it returns a vector, otherwise a scalar.
        """

        if len(X.shape) == 1 or X.shape[0] == 1:
            return self._constraint(X.ravel())

        constr = CArray.ones(X.shape[0])
        for i in xrange(0, X.shape[0]):
            constr[i] = self._constraint(X[i, :].ravel())  # todo: todense?
        return constr

    def projection(self, X):
        """
        Works either on a matrix, each row representing a data sample,
        or on a single (row) vector. Returns the projected data.
        """
        if len(X.shape) == 1 or X.shape[0] == 1:  # single row
            if self.is_violated(X) is True:
                self.logger.debug("Constraint violated, projecting...")
                X = self._projection(X.ravel())
            return X.ravel()

        X = X.deepcopy()  # Return a new array
        for i in xrange(X.shape[0]):
            x_i = X[i, :].ravel()
            if self.is_violated(x_i) is True:
                self.logger.debug("Constraint violated, projecting...")
                X[i, :] = self._projection(x_i.ravel())
        return X

    def gradient(self, X):
        """
        Works either on a matrix, each row representing a data sample,
        or on a single (row) vector. Returns the gradient of c(x) at x,
        in matrix form if X is a matrix.
        """
        if len(X.shape) == 1:
            X = self._gradient(X.ravel())
            return X.ravel()

        X = X.deepcopy()  # We return a new object
        for i in xrange(X.shape[0]):
            X[i, :] = self._gradient(X[i, :].ravel())
        return X
