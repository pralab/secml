"""
This module implements the generic constraint class
Implements equality/inequality constraints in the canonic form
    c(x) <= 0

.. moduleauthor:: Battista Biggio <battista.biggio@diee.unica.it>

"""
from abc import ABCMeta, abstractmethod
import six

from secml.core import CCreator
from secml.array import CArray


@six.add_metaclass(ABCMeta)
class CConstraint(CCreator):
    """Abstract class that defines basic methods for constraints."""
    __super__ = 'CConstraint'

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

    def is_violated(self, x, precision=4):
        """Returns the violated status of the constraint for each sample in x.

        Parameters
        ----------
        x : CArray
            Array of data, 1-D or 2-D, one sample for each row.
        precision : int, optional
            Number of digits to check when computing the violated status
            Default is 4.

        Returns
        -------
        CArray or bool
            Violated status of the constraint. Boolean True/False value.
            If input is 1-D or 2-D with shape[0] == 1, a bool is returned.
            If input is 2-D with shape[0] > 1, a 1-D boolean array
             is returned with the value of the constraint for each sample.

        """
        if x.ndim == 1 or x.shape[0] == 1:
            # Single point case
            return self._is_violated(x.ravel(), precision)

        # Multi-point case
        is_violated = CArray.ones(x.shape[0], dtype=bool)
        for i in xrange(x.shape[0]):
            is_violated[i] = self._is_violated(x[i, :].ravel(), precision)

        return is_violated

    def _is_violated(self, x, precision=4):
        """Returns the violated status of the constraint for the sample x.

        Parameters
        ----------
        x : CArray
            Flat 1-D array with the sample.
        precision : int, optional
            Number of digits to check when computing the violated status
            Default is 4.

        Returns
        -------
        bool
            Violated status of the constraint. Boolean True/False value.

        """
        if round(self._constraint(x), precision) > 0:
            return True
        return False

    @abstractmethod
    def _constraint(self, x):
        """Returns the value of the constraint for the sample x.

        Parameters
        ----------
        x : CArray
            Flat 1-D array with the sample.

        Returns
        -------
        scalar
            Value of the constraint.

        """
        raise NotImplementedError()

    def constraint(self, x):
        """Returns the value of the constraint for each sample in x.

        Parameters
        ----------
        x : CArray
            Array of data, 1-D or 2-D, one sample for each row.

        Returns
        -------
        CArray or scalar
            Value of the constraint.
            If input is 1-D or 2-D with shape[0] == 1, a scalar is returned.
            If input is 2-D with shape[0] > 1, a 1-D array is returned with
             the value of the constraint for each sample.

        """
        if x.ndim == 1 or x.shape[0] == 1:
            # Single point case
            return self._constraint(x.ravel())

        # Multi-point case
        constr = CArray.ones(x.shape[0], dtype=float)
        for i in xrange(x.shape[0]):
            constr[i] = self._constraint(x[i, :].ravel())
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

                # ensure that the projected point is within the feasible domain
                if self.is_violated(X.ravel()) is True:
                    raise RuntimeError(
                        "Projected x is outside of feasible domain!")

            return X.ravel()

        X = X.deepcopy()  # Return a new array
        for i in xrange(X.shape[0]):
            x_i = X[i, :].ravel()
            if self.is_violated(x_i) is True:
                self.logger.debug("Constraint violated, projecting...")
                X[i, :] = self._projection(x_i.ravel())
                
                # ensure that the projected point is within the feasible domain
                if self._is_violated(X[i, :].ravel()):
                    raise RuntimeError(
                        "Projected x is outside of feasible domain!")
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
