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
        self._center = center
        self._radius = radius
        return

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
        """Returns constraint L2 center."""
        return self._radius

    @radius.setter
    def radius(self, value):
        """Sets constraint L2 center."""
        self._radius = value

    # on a single sample
    def _constraint(self, x):
        """Return constraint."""
        return (x - self._center).norm() - self._radius

    def _projection(self, x):
        """Project x onto the feasible domain."""
        return self._center + self._radius * (x - self._center) / \
            (x - self._center).norm()

    def _gradient(self,x):
        return (x-self._center).ravel()/(x-self._center).norm(ord=2)