"""
.. module:: CConstraint
   :synopsis: Interface for equality/inequality constraints
                in the canonic form c(x) <= 0

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>

"""
from abc import ABCMeta, abstractmethod

from secml.core import CCreator
from secml.array import CArray


class CConstraint(CCreator, metaclass=ABCMeta):
    """Interface for equality/inequality constraints."""
    __super__ = 'CConstraint'

    def is_active(self, x, tol=1e-4):
        """Returns True if constraint is active.

        A constraint is active if c(x) = 0.

        By default we assume constraints of the form c(x) <= 0.

        Parameters
        ----------
        x : CArray
            Input sample.
        tol : float, optional
            Tolerance to use for comparing c(x) against 0. Default 1e-4.

        Returns
        -------
        bool
            True if constraint is active, False otherwise.

        """
        if not x.is_vector_like:
            raise ValueError("only a vector-like array is accepted")
        if abs(self._constraint(x)) <= tol:
            return True
        return False

    def is_violated(self, x):
        """Returns the violated status of the constraint for the sample x.

        We assume the constraint violated if c(x) <= 0.

        Parameters
        ----------
        x : CArray
            Input sample.

        Returns
        -------
        bool
            True if constraint is violated, False otherwise.

        """
        if not x.is_vector_like:
            raise ValueError("only a vector-like array is accepted")
        if self._constraint(x) > 0:
            return True
        return False

    @abstractmethod
    def _constraint(self, x):
        """Returns the value of the constraint for the sample x.

        Parameters
        ----------
        x : CArray
            Input sample.

        Returns
        -------
        float
            Value of the constraint.

        """
        raise NotImplementedError

    def constraint(self, x):
        """Returns the value of the constraint for the sample x.

        Parameters
        ----------
        x : CArray
            Input sample.

        Returns
        -------
        float
            Value of the constraint.

        """
        if not x.is_vector_like:
            raise ValueError("only a vector-like array is accepted")
        return float(self._constraint(x))

    @abstractmethod
    def _projection(self, x):
        """Project x onto feasible domain / within the given constraint.

        Parameters
        ----------
        x : CArray
            Input sample.

        Returns
        -------
        CArray
            Projected x onto feasible domain if constraint is violated.

        """
        raise NotImplementedError

    def projection(self, x):
        """Project x onto feasible domain / within the given constraint.

        If constraint is not violated by x, x is returned.

        Parameters
        ----------
        x : CArray
            Input sample.

        Returns
        -------
        CArray
            Projected x onto feasible domain if constraint is violated.
            Otherwise, x is returned as is.

        """
        if not x.is_vector_like:
            raise ValueError("only a vector-like array is accepted")
        if self.is_violated(x) is True:
            self.logger.debug("Constraint violated, projecting...")
            x = self._projection(x.ravel())
        return x.ravel()

    # This is not abstract as some constraints may not be differentiable
    def _gradient(self, x):
        """Returns the gradient of c(x) in x.

        Parameters
        ----------
        x : CArray
            Input sample.

        Returns
        -------
        CArray
            The gradient of the constraint computed on x.

        """
        raise NotImplementedError

    def gradient(self, x):
        """Returns the gradient of c(x) in x.

        Parameters
        ----------
        x : CArray
            Input sample.

        Returns
        -------
        CArray
            The gradient of the constraint computed on x.

        """
        if not x.is_vector_like:
            raise ValueError("only a vector-like array is accepted")
        return self._gradient(x).ravel()
