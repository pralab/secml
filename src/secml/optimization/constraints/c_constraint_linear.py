"""
.. module:: LinearConstraint
   :synopsis: Class that defines a linear constraint.

.. moduleauthor:: Battista Biggio <battista.biggio@diee.unica.it>
.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from secml.optimization.constraints import CConstraint
from secml.array import CArray


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
        """Returns the value of the constraint for the sample x.

        The constraint value y is given by:
         y = max(A.dot(x) - b)

        Parameters
        ----------
        x : CArray
            Flat 1-D array with the sample.

        Returns
        -------
        float
            Value of the constraint.

        """
        cons = CArray(self._A.dot(x.T)).ravel() - self._b
        return float(cons.max())

    def _projection(self, x):
        """Project x onto the convex polytope Ax-b <= 0."""
        # TODO: come si fa ? :)
        raise NotImplementedError()
