"""
.. module:: CConstraintL2
   :synopsis: L2 Constraint

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>

"""
from secml.optim.constraints import CConstraint
from secml.array import CArray


class CConstraintL2(CConstraint):
    """L2 Constraint.

    Parameters
    ----------
    center : scalar or CArray, optional
        Center of the constraint. Use an array to specify a different
        value for each dimension. Default 0.
    radius : scalar, optional
        The semidiagonal of the constraint. Default 1.

    Attributes
    ----------
    class_type : 'l2'

    """
    __class_type = 'l2'

    def __init__(self, center=0, radius=1):
        # Setting the value of the center (array or scalar)
        self.center = center
        # Setting the radius of the L2 ball (fixed)
        self.radius = radius

    @property
    def center(self):
        """Center of the constraint."""
        return self._center

    @center.setter
    def center(self, value):
        """Center of the constraint."""
        self._center = CArray(value)

    @property
    def radius(self):
        """Radius of the constraint."""
        return self._radius

    @radius.setter
    def radius(self, value):
        """Radius of the constraint."""
        self._radius = float(value)

    def _constraint(self, x):
        """Returns the value of the constraint for the sample x.

        The constraint value y is given by:
         y = ||x - center||_2 - radius

        Parameters
        ----------
        x : CArray
            Input sample.

        Returns
        -------
        float
            Value of the constraint.

        """
        return float((x - self.center).norm(order=2) - self.radius)

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
        # define tolerance and project onto radius-tol
        # to ensure that numerical errors do not violate the projection
        tol = 1e-6
        sub = (self._radius-tol) * (x - self.center)
        sub_l2 = (x - self.center).norm(order=2)
        if sub_l2 != 0:  # Avoid division by 0
            sub /= sub_l2
        out = self._center + sub
        return out.tosparse() if x.issparse else out

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
        sub = (x - self.center).ravel()
        # Avoid division by 0
        return sub if sub.norm() == 0 else sub / sub.norm()
