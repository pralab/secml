from secml.testing import CUnitTest

from secml.figure import CFigure
from secml.array import CArray
from secml.optim.function import CFunction
from secml.utils import fm


class CConstraintTestCases(CUnitTest):
    """Unittests interface for CConstraint."""

    def _test_is_active(self, c, p_in=None, p_out=None, p_on=None):
        """Test for CConstraint.is_active().

        Parameters
        ----------
        c : CConstraint
        p_in, p_out, p_on : CArray, optional
            3 points could be passes. One point inside the constraint,
            one point outside and one point on the constraint.

        """
        self.logger.info("Testing `.is_active` method for: {:}".format(c))

        def check_active(cons, point, expect):
            self.logger.info("Testing point: {:}".format(point))
            res = cons.is_active(point)
            self.assertEqual(expect, res)

        if p_in is None and p_out is None and p_on is None:
            raise ValueError("pass at least one point")

        if p_in is not None:
            # This point is INSIDE, constraint should NOT be active
            check_active(c, p_in, False)
            check_active(c, p_in.astype(int), False)

        if p_out is not None:
            # This point is OUTSIDE, constraint should NOT be active
            check_active(c, p_out, False)
            check_active(c, p_out.astype(int), False)

        if p_on is not None:
            # This point is ON, constraint SHOULD BE active
            check_active(c, p_on, True)
            check_active(c, p_on.astype(int), True)

    def _test_is_violated(self, c, p_in=None, p_out=None, p_on=None):
        """Test for CConstraint.is_violated().

        Parameters
        ----------
        c : CConstraint
        p_in, p_out, p_on : CArray, optional
            3 points are required. One point inside the constraint,
            one point outside and one point on the constraint.

        """
        self.logger.info("Testing `.is_violated` method for: {:}".format(c))

        def check_violated(cons, point, expect):
            self.logger.info("Testing point: {:}".format(point))
            res = cons.is_violated(point)
            self.assertEqual(expect, res)

        if p_in is None and p_out is None and p_on is None:
            raise ValueError("pass at least one point")

        if p_in is not None:
            # This point is INSIDE, constraint should NOT be violated
            check_violated(c, p_in, False)
            check_violated(c, p_in.astype(int), False)

        if p_on is not None:
            # This point is ON, constraint NOT be violated
            check_violated(c, p_on, False)
            check_violated(c, p_on.astype(int), False)

        if p_out is not None:
            # This point is OUTSIDE, constraint SHOULD BE violated
            check_violated(c, p_out, True)
            check_violated(c, p_out.astype(int), True)

    def _test_constraint(self, c, p_in=None, p_out=None, p_on=None):
        """Test for CConstraint.constraint().

        Parameters
        ----------
        c : CConstraint
        p_in, p_out, p_on : CArray, optional
            3 points are required. One point inside the constraint,
            one point outside and one point on the constraint.

        """
        self.logger.info("Testing `.constraint` method for: {:}".format(c))

        def check_constraint(cons, point, expect):
            res = cons.constraint(point)

            self.logger.info(
                ".constraint({:}): {:}".format(point, res))

            self.assertIsInstance(res, float)

            if expect == 'equal':
                self.assertEqual(0, res)
            elif expect == 'less':
                self.assertLess(res, 0)
            elif expect == 'greater':
                self.assertGreater(res, 0)
            else:
                raise ValueError(
                    "values {'equal', 'less', 'greater'} for `expect`")

        if p_in is None and p_out is None and p_on is None:
            raise ValueError("pass at least one point")

        if p_in is not None:
            # This point is INSIDE, constraint should be LESS then 0
            check_constraint(c, p_in, 'less')
            check_constraint(c, p_in.astype(int), 'less')

        if p_out is not None:
            # This point is OUTSIDE, constraint should be GREATER then 0
            check_constraint(c, p_out, 'greater')
            check_constraint(c, p_out.astype(int), 'greater')

        if p_on is not None:
            # This point is ON, constraint should be EQUAL to 0
            check_constraint(c, p_on, 'equal')
            check_constraint(c, p_on.astype(int), 'equal')

    def _test_projection(self, c, p_in=None, p_out=None,
                         p_on=None, p_out_expected=None):
        """Test for CConstraint.projection().

        Parameters
        ----------
        c : CConstraint
        p_in, p_out, p_on : CArray, optional
            3 points are required. One point inside the constraint,
            one point outside and one point on the constraint.
        p_out_expected : CArray, optional
            The expected value of .projection(p_out).

        """
        self.logger.info("Testing `.projection` method for: {:}".format(c))

        def check_projection(cons, point, expected):
            self.logger.info("Testing point: {:}".format(point))

            x_proj = cons.projection(point)

            self.logger.info("After projection: {:}".format(x_proj))
            self.logger.info("Expected: {:}".format(expected))

            self.assertIsInstance(x_proj, CArray)
            self.assertEqual(x_proj.issparse, point.issparse)

            # After projection, constraint should not be violated
            self.assertFalse(cons.is_violated(x_proj))

            self.assert_array_almost_equal(x_proj, expected, decimal=4)

        if p_in is None and p_out is None and p_on is None:
            raise ValueError("pass at least one point")

        if p_in is not None:
            # This point is INSIDE, should not change
            check_projection(c, p_in, p_in)
            check_projection(c, p_in.astype(int), p_in)

        if p_on is not None:
            # This point is ON, should not change
            check_projection(c, p_on, p_on)
            check_projection(c, p_on.astype(int), p_on)

        if p_out is not None:
            # This point is OUTSIDE, should be projected
            check_projection(c, p_out, p_out_expected)
            check_projection(c, p_out.astype(int), p_out_expected)

    def _test_plot(self, c, *points):
        """Visualize the constraint.

        Parameters
        ----------
        c : CConstraint
        *points : CArray
            A series of point to plot. Each point will be plotted before
            and after cosntraint projection.

        """
        self.logger.info("Plotting constrain {:}".format(c.class_type))

        grid_limits = [(-1, 2.5), (-1, 2.5)]

        fig = CFigure(height=6, width=6)

        fig.sp.plot_fun(func=c.constraint,
                        grid_limits=grid_limits,
                        n_grid_points=40,
                        levels=[0],
                        levels_linewidth=1.5)

        colors = ['g', 'r', 'c', 'm', 'y', 'k', 'w']
        for p_i, p in enumerate(points):
            self.logger.info(
                "Plotting point (color {:}): {:}".format(colors[p_i], p))
            fig.sp.scatter(*p, c=colors[p_i], zorder=10)
            p_proj = c.projection(p)
            self.logger.info(
                "Plotting point (color {:}): {:}".format(colors[p_i], p_proj))
            fig.sp.scatter(*p_proj, c=colors[p_i], zorder=10)

        filename = "test_constraint_{:}.pdf".format(c.class_type)

        fig.savefig(fm.join(fm.abspath(__file__), filename))

    def _test_gradient(self, c, p, th=1e-6):
        """Compare the analytical with the numerical gradient.

        Parameters
        ----------
        c : CConstraint
        p : CArray
            The point on which the gradient is computed.
        th : float
            Tolerance for approximation check.

        """
        self.logger.info("Testing `.gradient({:})` for {:}".format(p, c))
        gradient = c.gradient(p)

        self.assertTrue(gradient.is_vector_like)
        self.assertEqual(p.size, gradient.size)

        # Numerical gradient
        num_gradient = CFunction(c.constraint).approx_fprime(p, 1e-8)

        # Compute the norm of the difference
        error = (gradient - num_gradient).norm()

        self.logger.info("Analytic grad: {:}".format(gradient))
        self.logger.info("Numeric grad: {:}".format(num_gradient))

        self.logger.info("norm(grad - num_grad): {:}".format(error))
        self.assertLess(error, th)

        self.assertIsSubDtype(gradient.dtype, float)
