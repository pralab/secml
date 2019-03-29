"""
.. module:: CConstraintBox
   :synopsis: Box constraint.

.. moduleauthor:: Battista Biggio <battista.biggio@diee.unica.it>

"""
import numpy as np
from secml.optim.constraints import CConstraint
from secml.array import CArray
from secml.core.constants import inf


class CConstraintBox(CConstraint):
    """Class that defines a box constraint.

    Attributes
    ----------
    class_type : 'box'

    """
    __class_type = 'box'

    def __init__(self, lb=None, ub=None):
        self.lb = lb
        self.ub = ub

    @property
    def lb(self):
        """Lower bound."""
        return self._lb

    @lb.setter
    def lb(self, value):
        """Lower bound."""
        value = -inf if value is None else value
        self._lb = CArray(value).ravel()

    @property
    def ub(self):
        """Upper bound."""
        return self._ub

    @ub.setter
    def ub(self, value):
        """Upper bound."""
        value = inf if value is None else value
        self._ub = CArray(value).ravel()

    @property
    def center(self):
        """Center of the constraint."""
        # FIXME: WORKAROUND FOR RUNTIMEWARNING inf + inf
        if any(np.isinf(self.ub.tondarray())) or \
                any(np.isinf(self.lb.tondarray())):
            if self.ub.size > self.lb.size:
                return CArray.empty(shape=self.ub.shape) * np.nan
            else:
                return CArray.empty(shape=self.lb.shape) * np.nan
        return CArray(0.5 * (self.ub + self.lb)).ravel()

    @property
    def radius(self):
        """Radius of the constraint."""
        # FIXME: WORKAROUND FOR RUNTIMEWARNING inf + inf
        if any(np.isinf(self.ub.tondarray())) or \
                any(np.isinf(self.lb.tondarray())):
            if self.ub.size > self.lb.size:
                return CArray.empty(shape=self.ub.shape) * np.nan
            else:
                return CArray.empty(shape=self.lb.shape) * np.nan
        return CArray(0.5 * (self.ub - self.lb)).ravel()

    def set_center_radius(self, c, r):
        """Set constraint bounds in terms of center and radius.

        This method will transform the input center/radius as follows:
          lb = center - radius
          ub = center + radius

        Parameters
        ----------
        c : scalar
            Constraint center.
        r : scalar
            Constraint radius.

        """
        self.lb = c - r
        self.ub = c + r

    def _constraint(self, x):
        """Returns the value of the constraint for the sample x.

        The constraint value y is given by:
         y = max(abs(x - center) - radius)

        Parameters
        ----------
        x : CArray
            Input array.

        Returns
        -------
        float
            Value of the constraint.

        """
        # if x is sparse, and center and radius are not (sparse) vectors
        if x.issparse and self.center.size != x.size and \
                self.radius.size != x.size:
            return self._constraint_sparse(x)

        c = self.center
        r = self.radius

        # FIXME: WORKAROUND FOR RUNTIMEWARNING x - nan
        if any(np.isnan(c.tondarray())) or any(np.isnan(r.tondarray())):
            return np.nan

        return float((abs(x - c) - r).max())

    def _constraint_sparse(self, x):
        """Returns the value of the constraint for the sample x.

        This implementation for sparse arrays only allows a scalar value
         for center and radius.

        Parameters
        ----------
        x : CArray
            Input array.

        Returns
        -------
        float
            Value of the constraint.

        """
        if self.center.size > 1 and self.radius.size > 1:
            raise ValueError("Box center and radius are not scalar values.")

        m0 = (abs(0 - self.center) - self.radius).max()
        if x.nnz == 0:
            return float(m0)

        # computes constraint values (l-inf dist. to center) for nonzero values
        z = abs(CArray(x.nnz_data).todense() - self.center) - self.radius
        m = z.max()
        # if there are no zeros in x... (it may be effectively "dense")
        if x.nnz == x.size:
            # return current maximum value
            return float(m)

        # otherwise evaluate also the l-inf dist. of 0 elements to the center,
        # and also consider that in the max computation
        return float(max(m, m0))

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
        # If bound is float, ensure x is float
        if np.issubdtype(self.ub.dtype, np.floating) or \
                np.issubdtype(self.ub.dtype, np.floating):
            x = x.astype(float)

        if self.ub.size == 1:  # Same ub for all the features
            x[x >= self._ub] = self._ub
        else:
            x[x >= self._ub] = self._ub[x >= self._ub]

        if self.lb.size == 1:  # Same lb for all the features
            x[x <= self._lb] = self._lb
        else:
            x[x <= self._lb] = self._lb[x <= self._lb]

        return x
