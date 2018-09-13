"""
.. module:: LinearConstraint
   :synopsis: Class that defines a linear constraint.

.. moduleauthor:: Battista Biggio <battista.biggio@diee.unica.it>
.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from prlib.optimization.constraints import CConstraint
from prlib.array import CArray


class CConstraintLinear(CConstraint):
    """Class that defines a linear constraint in the form Ax <= b."""

    class_type = "linear"

    def __init__(self, A, b):

        super(CConstraintLinear, self).__init__()

        if A is None or b is None:
            raise ValueError("None values not allowed for A, b")

        self.A = A
        self.b = b

    @property
    def A(self):
        """Returns upper bound."""
        return self._A

    @A.setter
    def A(self, value):
        """Sets A."""
        self._A = CArray(value)

    @property
    def b(self):
        """Returns lower bound."""
        return self._b

    @b.setter
    def b(self, value):
        self._b = CArray(CArray(value).ravel())

    def _constraint(self, x):
        """
        This function returns the value of Ax-b,
        to be compared against 0.
        x is assumed to be a row (flat) vector
        """
        cons = (self._A.dot(x.T)).ravel() - self._b
        return float(cons.max())

    def _projection(self, x):
        """Project x onto the convex polytope Ax-b <= 0."""
        # TODO: come si fa ? :)
        raise NotImplementedError()
