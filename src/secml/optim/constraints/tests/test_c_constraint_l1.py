from secml.optim.constraints.tests import CConstraintTestCases

from math import acos

from secml.optim.constraints import CConstraintL1
from secml.array import CArray


class TestCConstraintL1(CConstraintTestCases):
    """Unittest for CConstraintL1."""

    def setUp(self):

        self.c = CConstraintL1(center=1, radius=1)
        self.c_array = CConstraintL1(center=CArray([1, 1]), radius=1)

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
        self._test_is_violated(
            self.c_array, self.p1_inside.tosparse(),
            self.p2_outside.tosparse(), self.p3_on.tosparse())

    def test_is_violated(self):
        """Test for CConstraint.is_violated()."""
        self._test_is_violated(
            self.c, self.p1_inside, self.p2_outside, self.p3_on)
        self._test_is_violated(
            self.c_array, self.p1_inside, self.p2_outside, self.p3_on)

        # Test for sparse arrays, works only for a center defined as CArray
        self._test_is_active(
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
                              self.p3_on, CArray([1.5, 1.5]))
        self._test_projection(self.c_array, self.p1_inside, self.p2_outside,
                              self.p3_on, CArray([1.5, 1.5]))

        # Test for sparse arrays, works only for a center defined as CArray
        self._test_projection(
            self.c_array, self.p1_inside.tosparse(),
            self.p2_outside.tosparse(), self.p3_on.tosparse(),
            CArray([1.5, 1.5], tosparse=True))

    def test_gradient(self):
        """Test for CConstraint.gradient()."""
        # [1. 1.] is the center of the constraint, expected grad [0, c0]
        # however, numerical gradient is struggling so we avoid its omparison
        self.assert_array_almost_equal(
            self.c.gradient(self.p1_inside), CArray([0, 0]))

        self._test_gradient(self.c, CArray([1.1, 1.2]))
        self._test_gradient(self.c, self.p2_outside)

        # [0. 1.] is the verge of the constraint, expected grad [-1, 1]
        # however, numerical gradient is struggling so we avoid its comparison
        self.assert_array_almost_equal(
            self.c.gradient(self.p3_on), CArray([-1., 0.]))

    def test_subgradient(self):
        """Check if the subgradient is computed correctly

        Subgradient should lie in the cone made up by the subgradients.

        """
        c = CConstraintL1(center=0, radius=1)

        x0 = CArray([0, 1])

        p_min = CArray([1, 1])
        p_max = CArray([-1, 1])

        gradient = c.gradient(x0)

        # normalize the points
        norm_center = x0 / x0.norm(2)
        norm_p_min = p_min / p_min.norm(2)
        norm_p_max = p_max / p_max.norm(2)
        norm_gradient = gradient / gradient.norm(2)

        angl1 = round(acos(norm_center.dot(norm_gradient)), 5)
        angl2 = round(acos(norm_p_min.dot(norm_p_max)) / 2.0, 5)

        self.logger.info("Subgrad in {:} is:\n{:}".format(x0, gradient))

        self.assertLessEqual(angl1, angl2, "Subgrad is not inside the cone of "
                                           "{:} and {:}".format(p_min, p_max))

    def test_plot(self):
        """Visualize the constraint."""
        # Plotting constraint and "critical" points
        self._test_plot(
            self.c, self.p1_inside, self.p2_outside, self.p3_on)


if __name__ == '__main__':
    CConstraintTestCases.main()
