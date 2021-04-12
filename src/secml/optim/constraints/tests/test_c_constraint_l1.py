from secml.optim.constraints.tests import CConstraintTestCases

from math import acos

from secml.optim.constraints import CConstraintL1
from secml.array import CArray


class TestCConstraintL1(CConstraintTestCases):
    """Unittest for CConstraintL1."""

    def setUp(self):

        self.c0 = CConstraintL1()  # center=0, radius=1
        self.c = CConstraintL1(center=1, radius=1)
        self.c_array = CConstraintL1(center=CArray([1, 1]), radius=1)

        # create a point that lies inside the constraints
        self.c0_p1_inside = CArray([0., 0.])
        self.c_p1_inside = CArray([1., 1.])
        # create a point that lies outside the constraints
        self.c0_p2_outside = CArray([1., 1.])
        self.c_p2_outside = CArray([2., 2.])
        # create a point that lies on the constraints
        self.c0_p3_on = CArray([0., 1.])
        self.c_p3_on = CArray([0., 1.])

    def test_is_active(self):
        """Test for CConstraint.is_active()."""
        self._test_is_active(
            self.c0, self.c0_p1_inside, self.c0_p2_outside, self.c0_p3_on)
        self._test_is_active(
            self.c, self.c_p1_inside, self.c_p2_outside, self.c_p3_on)
        self._test_is_active(
            self.c_array, self.c_p1_inside, self.c_p2_outside, self.c_p3_on)

        # Test for sparse arrays
        self._test_is_violated(
            self.c0,
            self.c0_p1_inside.tosparse(),
            self.c0_p2_outside.tosparse(),
            self.c0_p3_on.tosparse())
        self._test_is_violated(
            self.c_array,
            self.c_p1_inside.tosparse(),
            self.c_p2_outside.tosparse(),
            self.c_p3_on.tosparse())

    def test_is_violated(self):
        """Test for CConstraint.is_violated()."""
        self._test_is_violated(
            self.c0, self.c0_p1_inside, self.c0_p2_outside, self.c0_p3_on)
        self._test_is_violated(
            self.c, self.c_p1_inside, self.c_p2_outside, self.c_p3_on)
        self._test_is_violated(
            self.c_array, self.c_p1_inside, self.c_p2_outside, self.c_p3_on)

        # Test for sparse arrays
        self._test_is_active(
            self.c0,
            self.c0_p1_inside.tosparse(),
            self.c0_p2_outside.tosparse(),
            self.c0_p3_on.tosparse())
        self._test_is_active(
            self.c_array,
            self.c_p1_inside.tosparse(),
            self.c_p2_outside.tosparse(),
            self.c_p3_on.tosparse())

    def test_constraint(self):
        """Test for CConstraint.constraint()."""
        self._test_constraint(
            self.c0, self.c0_p1_inside, self.c0_p2_outside, self.c0_p3_on)
        self._test_constraint(
            self.c, self.c_p1_inside, self.c_p2_outside, self.c_p3_on)
        self._test_constraint(
            self.c_array, self.c_p1_inside, self.c_p2_outside, self.c_p3_on)

        # Test for sparse arrays
        self._test_constraint(
            self.c0,
            self.c0_p1_inside.tosparse(),
            self.c0_p2_outside.tosparse(),
            self.c0_p3_on.tosparse())
        self._test_constraint(
            self.c_array,
            self.c_p1_inside.tosparse(),
            self.c_p2_outside.tosparse(),
            self.c_p3_on.tosparse())

    def test_projection(self):
        """Test for CConstraint.projection()."""
        self._test_projection(self.c0, self.c0_p1_inside, self.c0_p2_outside,
                              self.c0_p3_on, CArray([0.5, 0.5]))
        self._test_projection(self.c, self.c_p1_inside, self.c_p2_outside,
                              self.c_p3_on, CArray([1.5, 1.5]))
        self._test_projection(self.c_array, self.c_p1_inside, self.c_p2_outside,
                              self.c_p3_on, CArray([1.5, 1.5]))

        # Test for sparse arrays
        self._test_projection(
            self.c0, self.c0_p1_inside.tosparse(),
            self.c0_p2_outside.tosparse(), self.c0_p3_on.tosparse(),
            CArray([0.5, 0.5], tosparse=True))
        self._test_projection(
            self.c_array, self.c_p1_inside.tosparse(),
            self.c_p2_outside.tosparse(), self.c_p3_on.tosparse(),
            CArray([1.5, 1.5], tosparse=True))

    def test_gradient(self):
        """Test for CConstraint.gradient()."""
        # [0. 0.] is the center of the constraint, expected grad [0, c0]
        # however, numerical gradient is struggling so we avoid its comparison
        self.assert_array_almost_equal(
            self.c0.gradient(self.c0_p1_inside), CArray([0, 0]))

        # [1. 1.] is the center of the constraint, expected grad [0, c0]
        # however, numerical gradient is struggling so we avoid its comparison
        self.assert_array_almost_equal(
            self.c.gradient(self.c_p1_inside), CArray([0, 0]))

        self._test_gradient(self.c0, CArray([0.1, 0.2]))
        self._test_gradient(self.c0, self.c0_p2_outside)

        self._test_gradient(self.c, CArray([1.1, 1.2]))
        self._test_gradient(self.c, self.c_p2_outside)

        # [0. 1.] is the verge of the constraint, expected grad [0, 1]
        # however, numerical gradient is struggling so we avoid its comparison
        self.assert_array_almost_equal(
            self.c0.gradient(self.c0_p3_on), CArray([0., 1.]))
        # [0. 1.] is the verge of the constraint, expected grad [-1, 1]
        # however, numerical gradient is struggling so we avoid its comparison
        self.assert_array_almost_equal(
            self.c.gradient(self.c_p3_on), CArray([-1., 0.]))

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
        self._test_plot(self.c0,
                        self.c0_p1_inside,
                        self.c0_p2_outside,
                        self.c0_p3_on,
                        label='c0')
        self._test_plot(self.c,
                        self.c_p1_inside,
                        self.c_p2_outside,
                        self.c_p3_on,
                        label='c')


if __name__ == '__main__':
    CConstraintTestCases.main()
