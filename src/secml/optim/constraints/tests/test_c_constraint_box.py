from secml.optim.constraints.tests import CConstraintTestCases

from secml.optim.constraints import CConstraintBox
from secml.array import CArray


class TestConstraintBox(CConstraintTestCases):
    """Unittest for CConstraintBox."""

    def setUp(self):

        self.c = CConstraintBox(lb=CArray([0, -0.5]), ub=1.5)

        # create a point that lies inside the constraint
        self.p1_inside = CArray([1., 1.])
        # create a point that lies outside the constraint
        self.p2_outside = CArray([2., 2.])
        # create a point that lies on the constraint
        self.p3_on = CArray([0., 1.])

    def test_is_active(self):
        """Test for CConstraint.is_active()."""
        self._test_is_active(
            self.c, self.p1_inside, self.p2_outside, self.p3_on)

        # Test for sparse arrays
        self._test_is_active(
            self.c, self.p1_inside.tosparse(),
            self.p2_outside.tosparse(), self.p3_on.tosparse())

    def test_is_violated(self):
        """Test for CConstraint.is_violated()."""
        self._test_is_violated(
            self.c, self.p1_inside, self.p2_outside, self.p3_on)

        # Test for sparse arrays
        self._test_is_active(
            self.c, self.p1_inside.tosparse(),
            self.p2_outside.tosparse(), self.p3_on.tosparse())

    def test_constraint(self):
        """Test for CConstraint.constraint()."""
        self._test_constraint(
            self.c, self.p1_inside, self.p2_outside, self.p3_on)

        # Test for sparse arrays
        self._test_constraint(
            self.c, self.p1_inside.tosparse(),
            self.p2_outside.tosparse(), self.p3_on.tosparse())

    def test_projection(self):
        """Test for CConstraint.projection()."""
        self._test_projection(self.c, self.p1_inside, self.p2_outside,
                              self.p3_on, CArray([1.5, 1.5]))

        # Test for sparse arrays
        self._test_projection(
            self.c, self.p1_inside.tosparse(),
            self.p2_outside.tosparse(), self.p3_on.tosparse(),
            CArray([1.5, 1.5], tosparse=True))

        # Check sparse arrays and scalar ub/lb
        # (corner case of sparse arrays as we cannot sub/add scalara)
        c = CConstraintBox(lb=0, ub=1.5)
        self._test_projection(
            c, CArray([1., 1., 0.], tosparse=True),
            CArray([2., 2., 0.], tosparse=True),
            CArray([0., 1., 0.]).tosparse(),
            CArray([1.5, 1.5, 0], tosparse=True))

    def test_plot(self):
        """Visualize the constraint."""
        # Plotting constraint and "critical" points
        self._test_plot(
            self.c, self.p1_inside, self.p2_outside, self.p3_on)


if __name__ == '__main__':
    CConstraintTestCases.main()
