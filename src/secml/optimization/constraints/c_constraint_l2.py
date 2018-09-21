"""
C_constraint
@author: Battista Biggio
@author: Paolo Russu

This module contains the class for the L2 constraint

"""
from secml.optimization.constraints import CConstraint


class CConstraintL2(CConstraint):

    class_type = "l2"

    def __init__(self, center=0, radius=1):
        # Setting the value of the center (array or scalar)
        self._center = None
        self.center = center
        # Setting the radius of the L2 ball (fixed)
        self._radius = None
        self.radius = radius

    @property
    def center(self):
        """Returns constraint L2 center."""
        return self._center

    @center.setter
    def center(self, value):
        """Sets constraint L2 center."""
        self._center = value

    @property
    def radius(self):
        """Returns constraint L2 radius."""
        return self._radius

    @radius.setter
    def radius(self, value):
        """Sets constraint L2 radius."""
        self._radius = float(value)

    def _constraint(self, x):
        """Returns the value of the constraint for the sample x.

        The constraint value y is given by:
         y = ||x - center||_2 - radius

        Parameters
        ----------
        x : CArray
            Flat 1-D array with the sample.

        Returns
        -------
        float
            Value of the constraint.

        """
        return float((x - self._center).norm() - self._radius)

    def _projection(self, x):
        """Project x onto the feasible domain."""
        return self._center + self._radius * (x - self._center) / \
            (x - self._center).norm()

    def _gradient(self,x):
        return (x-self._center).ravel()/(x-self._center).norm(ord=2)