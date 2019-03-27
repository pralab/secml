"""
.. module:: CConstraintL2
   :synopsis: L2 Constraint

.. moduleauthor:: Battista Biggio <battista.biggio@diee.unica.it>

"""
from secml.optim.constraints import CConstraint


class CConstraintL2(CConstraint):
    """L2 Constraint.

    Attributes
    ----------
    class_type : 'l2'

    """
    __class_type = 'l2'

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
        return float((x - self._center).norm(order=2) - self._radius)

    def _projection(self, x):
        """Project x onto the feasible domain."""
        sub = self._radius * (x - self._center)
        sub_l2 = (x - self._center).norm(order=2)
        if sub_l2 != 0:  # Avoid division by 0
            sub /= sub_l2
        return self._center + sub

    def _gradient(self, x):
        """Returns the gradient of the constraint function at x."""
        sub = (x - self._center).ravel()
        # Avoid division by 0
        return sub if sub.norm() == 0 else sub / sub.norm()
