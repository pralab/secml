"""
.. module:: CConstraintBox
   :synopsis: Box constraint.

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
import numpy as np
from secml.optim.constraints import CConstraint
from secml.array import CArray
from secml.core.constants import inf


class CConstraintBox(CConstraint):
    """Class that defines a box constraint.

    Parameters
    ----------
    lb, ub : scalar or CArray or None, optional
        Bounds of the constraints.
        If scalar, the same bound will be applied to all features.
        If CArray, should contain a bound for each feature.
        If None, a +/- inf ub/lb bound will be used for all features.

    Attributes
    ----------
    class_type : 'box'

    """
    __class_type = 'box'

    def __init__(self, lb=None, ub=None):

        # Lower bound
        lb = -inf if lb is None else lb
        self._lb = lb.ravel() if isinstance(lb, CArray) else lb
        # Upper bound
        ub = inf if ub is None else ub
        self._ub = ub.ravel() if isinstance(ub, CArray) else ub

        self._validate_bounds()  # Check if bounds have been correctly defined

    @property
    def lb(self):
        """Lower bound."""
        return self._lb

    @property
    def ub(self):
        """Upper bound."""
        return self._ub

    def _validate_bounds(self):
        """Check that bounds are valid.

        Must:
         - be lb <= ub
         - have same size if both CArray.

        """
        lb_array = CArray(self.lb)
        ub_array = CArray(self.ub)

        if isinstance(self.lb, CArray) and isinstance(self.ub, CArray):
            if lb_array.size != ub_array.size:
                raise ValueError("`ub` and `lb` must have the same size if "
                                 "both `CArray`. Currently {:} and {:}"
                                 "".format(ub_array.size, lb_array.size))

        if (lb_array > ub_array).any():
            raise ValueError("`lb` must be lower or equal than `ub`")

    def _check_inf(self):
        """Return True if any of the bounds are or contain inf.

        Returns
        -------
        bool

        """
        # Convert both bounds to CArray for simplicity
        if CArray(self.ub).is_inf().any() or CArray(self.lb).is_inf().any():
            return True
        return False

    @property
    def center(self):
        """Center of the constraint."""
        if self._check_inf() is True:
            raise ValueError("cannot compute `center` as at least one value "
                             "in the bounds is +/- `inf`")
        return CArray(0.5 * (self.ub + self.lb)).ravel()

    @property
    def radius(self):
        """Radius of the constraint."""
        if self._check_inf() is True:
            raise ValueError("cannot compute `radius` as at least one value "
                             "in the bounds is +/- `inf`")
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
        self._lb = c - r
        self._ub = c + r

        self._validate_bounds()  # Check if bounds have been correctly defined

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
        # If at least one value in the bounds is +/- inf,
        # the constraint is never active
        if self._check_inf() is True:
            return False

        return super(CConstraintBox, self).is_active(x, tol=tol)

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
        return (x < self.lb).logical_or(x > self.ub).any()

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

        return float((abs(x - self.center) - self.radius).max())

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
        if np.issubdtype(CArray(self.ub).dtype, np.floating) or \
                np.issubdtype(CArray(self.ub).dtype, np.floating):
            x = x.astype(float)

        if isinstance(self.ub, CArray):
            x[x > self.ub] = self.ub[x > self.ub]
        else:  # Same ub for all the features
            x[x > self.ub] = self.ub

        if isinstance(self.lb, CArray):
            x[x < self.lb] = self.lb[x < self.lb]
        else:  # Same lb for all the features
            x[x < self.lb] = self.lb

        return x
