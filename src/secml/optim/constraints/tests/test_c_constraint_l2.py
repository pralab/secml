from secml.optim.constraints.tests import CConstraintTestCases

from secml.optim.constraints import CConstraintL2
from secml.array import CArray


class TestCConstraintL2(CConstraintTestCases):
    """Unittest for CConstraintL2."""

    def setUp(self):
        self.c = CConstraintL2(center=1, radius=1)
        self.c_array = CConstraintL2(center=CArray([1, 1]), radius=1)

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
        self._test_is_active(
            self.c_array, self.p1_inside, self.p2_outside, self.p3_on)

        # Test for sparse arrays, works only for a center defined as CArray
        self._test_is_active(
            self.c_array, self.p1_inside.tosparse(),
            self.p2_outside.tosparse(), self.p3_on.tosparse())

    def test_is_violated(self):
        """Test for CConstraint.is_violated()."""
        self._test_is_violated(
            self.c, self.p1_inside, self.p2_outside, self.p3_on)
        self._test_is_violated(
            self.c_array, self.p1_inside, self.p2_outside, self.p3_on)

        # Test for sparse arrays, works only for a center defined as CArray
        self._test_is_violated(
            self.c_array, self.p1_inside.tosparse(),
            self.p2_outside.tosparse(), self.p3_on.tosparse())

    def test_constraint(self):
        """Test for CConstraint.constraint()."""
        self._test_constraint(
            self.c, self.p1_inside, self.p2_outside, self.p3_on)
        self._test_constraint(
            self.c_array, self.p1_inside, self.p2_outside, self.p3_on)

        # Test for sparse arrays, works only for a center defined as CArray
        self._test_constraint(
            self.c_array, self.p1_inside.tosparse(),
            self.p2_outside.tosparse(), self.p3_on.tosparse())

    def test_projection(self):
        """Test for CConstraint.projection()."""
        self._test_projection(self.c, self.p1_inside, self.p2_outside,
                              self.p3_on, CArray([1.7071, 1.7071]))
        self._test_projection(self.c_array, self.p1_inside, self.p2_outside,
                              self.p3_on, CArray([1.7071, 1.7071]))

        # Test for sparse arrays, works only for a center defined as CArray
        self._test_projection(
            self.c_array, self.p1_inside.tosparse(),
            self.p2_outside.tosparse(), self.p3_on.tosparse(),
            CArray([1.7071, 1.7071], tosparse=True))

    def test_gradient(self):
        """Test for CConstraint.gradient()."""
        # [1. 1.] is the center of the constraint, expected grad [0, 0]
        # however, numerical gradient is struggling so we avoid its comparison
        self.assert_array_almost_equal(
            self.c.gradient(self.p1_inside), CArray([0, 0]))

        self._test_gradient(self.c, CArray([1.1, 1.2]))
        self._test_gradient(self.c, self.p2_outside)
        self._test_gradient(self.c, self.p3_on)

    def test_plot(self):
        """Visualize the constraint."""
        # Plotting constraint and "critical" points
        self._test_plot(
            self.c, self.p1_inside, self.p2_outside, self.p3_on)

    def test_projection_and_violated(self):
        """Test that projection returns a point within the domain, even when
        numerical errors may create problems."""
        p = CArray([-10.00000053, 100000.432235252352]) + 2.32323
        p = self.c.projection(p)
        self.assertFalse(self.c.is_violated(p))


if __name__ == '__main__':
    CConstraintTestCases.main()
