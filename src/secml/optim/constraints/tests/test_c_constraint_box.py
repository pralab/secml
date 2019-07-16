from secml.optim.constraints.tests import CConstraintTestCases

from secml.optim.constraints import CConstraintBox
from secml.array import CArray
from secml.core.constants import inf


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

    def test_check_bounds(self):
        """Check validation of lb/ub"""
        CConstraintBox(lb=0.5, ub=1.5)  # Fine
        CConstraintBox(lb=CArray([0, -0.5]), ub=1.5)  # Fine
        CConstraintBox(lb=-1.5, ub=CArray([inf, -0.5]))  # Fine
        CConstraintBox(lb=CArray([-inf, 0]), ub=CArray([0, 0.5]))  # Fine

        # LB > UB
        with self.assertRaises(ValueError):
            CConstraintBox(lb=2, ub=1.5)
        with self.assertRaises(ValueError):
            CConstraintBox(lb=1.5, ub=CArray([2, -0.5]))
        with self.assertRaises(ValueError):
            CConstraintBox(lb=CArray([2, 0]), ub=CArray([-1.5, 1.5]))

        # LB, UB both CArray but with different dimensions
        with self.assertRaises(ValueError):
            CConstraintBox(lb=CArray([0]), ub=CArray([-1.5, 1.5]))

    def test_is_active(self):
        """Test for CConstraint.is_active()."""
        self._test_is_active(
            self.c, self.p1_inside, self.p2_outside, self.p3_on)

        # Test for sparse arrays
        self._test_is_active(
            self.c, self.p1_inside.tosparse(),
            self.p2_outside.tosparse(), self.p3_on.tosparse())

        # Constraint with one or more inf, should be never active
        c = CConstraintBox(lb=CArray([0, -0.5]), ub=inf)
        self.assertEqual(c.is_active(self.p1_inside), False)
        self.assertEqual(c.is_active(self.p2_outside), False)
        self.assertEqual(c.is_active(self.p3_on), False)

        c = CConstraintBox(lb=CArray([0, -inf]), ub=2)
        self.assertEqual(c.is_active(self.p1_inside), False)
        self.assertEqual(c.is_active(self.p2_outside), False)
        self.assertEqual(c.is_active(self.p3_on), False)

    def test_is_violated(self):
        """Test for CConstraint.is_violated()."""
        self._test_is_violated(
            self.c, self.p1_inside, self.p2_outside, self.p3_on)

        # Test for sparse arrays
        self._test_is_violated(
            self.c, self.p1_inside.tosparse(),
            self.p2_outside.tosparse(), self.p3_on.tosparse())

        # Constraint with one or more inf
        c = CConstraintBox(lb=CArray([0, -inf]), ub=1.5)
        self._test_is_violated(
            c, self.p1_inside, self.p2_outside, self.p3_on)

        c = CConstraintBox(lb=CArray([0, -0.5]), ub=inf)
        self._test_is_violated(  # Using [-2, -2] as outside
            c, self.p1_inside, -self.p2_outside, self.p3_on)

    def test_constraint(self):
        """Test for CConstraint.constraint()."""
        self._test_constraint(
            self.c, self.p1_inside, self.p2_outside, self.p3_on)

        # Test for sparse arrays
        self._test_constraint(
            self.c, self.p1_inside.tosparse(),
            self.p2_outside.tosparse(), self.p3_on.tosparse())

        # Constraint with one or more inf, error should be raised
        c = CConstraintBox(lb=CArray([0, -inf]), ub=1.5)
        with self.assertRaises(ValueError):
            c.constraint(self.p1_inside)

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

        # Constraint with one or more inf
        c = CConstraintBox(lb=CArray([0, -inf]), ub=1.5)
        self._test_projection(
            c, self.p1_inside, self.p2_outside,
            p_out_expected=CArray([1.5, 1.5]))

        c = CConstraintBox(lb=CArray([-inf, -0.5]), ub=inf)
        self._test_projection(  # Using [-2, -2] as outside, expect [-2, -0.5]
            c, self.p1_inside, -self.p2_outside,
            p_out_expected=CArray([-2, -0.5]))

    def test_plot(self):
        """Visualize the constraint."""
        # Plotting constraint and "critical" points
        self._test_plot(
            self.c, self.p1_inside, self.p2_outside, self.p3_on)


if __name__ == '__main__':
    CConstraintTestCases.main()
