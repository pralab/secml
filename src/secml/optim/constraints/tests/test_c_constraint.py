from abc import ABCMeta, abstractmethod

from numpy import *

from secml.utils import CUnitTest
from secml.figure import CFigure
from secml.array import CArray
from secml.optim.function import CFunction


class CConstraintTestCases(object):
    """Wrapper for TestCConstraint to make unittest.main() work correctly."""

    class TestCConstraint(CUnitTest):
        """Unit test for CConstraint."""
        __metaclass__ = ABCMeta

        @abstractmethod
        def _constr_creation(self):
            raise NotImplementedError

        @abstractmethod
        def _set_constr_name(self):
            raise NotImplementedError

        def _create_test_points(self):
            # create a point that lies inside the constraint
            self._p1_inside = CArray([0, 0])
            # create a point that lies outside the constraint
            self._p2_outside = CArray([2, 2])
            # create a point that lies on the constraint
            self._p3_on = CArray([0, 1])

        def test_plot_constraint_visualization(self):
            grid_limits = [(-3, 3), (-3, 3)]
            fig = CFigure(height=6, width=6)
            fig.switch_sptype(sp_type='function')
            fig.sp.plot_fobj(func=self._constr.constraint,
                             grid_limits=grid_limits)
            fig.sp.plot_fobj(func=self._constr.constraint,
                             plot_background=False,
                             n_grid_points=40,
                             grid_limits=grid_limits,
                             levels=[0])

            file_name = "test_{:}_constraint.pdf".format(self._constr_name)

            fig.savefig(file_name)

        def test_is_active(self):
            p1_active = self._constr.is_active(self._p1_inside)
            p2_active = self._constr.is_active(self._p2_outside)
            p3_active = self._constr.is_active(self._p3_on)

            self.assertLessEqual(p1_active, False, "The point lies "
                                                     "inside the constraint, therefore the value of the funciton is "
                                                     "active should be False")
            self.assertLessEqual(p2_active, False, "The point lies "
                                                     "inside the constraint, "
                                                     "therefore the value of the funciton is "
                                                     "active should be False")
            self.assertLessEqual(p3_active, True, "The point lies "
                                                    "on the constraint, "
                                                    "therefore the value of the funciton is "
                                                    "active should be "
                                                    "True")

        def _test_is_violated(self):
            """
            Check the `is_violated` function of the constraint object.
            (It should return True if the point lies outside the
            constraint and False otherwise).
            """
            p1_violated = self._constr.is_violated(self._p1_inside)
            p2_violated = self._constr.is_violated(self._p2_outside)
            p3_violated = self._constr.is_violated(self._p3_on)

            self.assertLessEqual(p1_violated, False, "The point lies "
                                                     "inside the constraint therefore the value of the funciton is "
                                                     "violated should be False")
            self.assertLessEqual(p2_violated, True, "The point lies "
                                                    "inside the constraint therefore the value of the funciton is "
                                                    "violated should be True")
            self.assertLessEqual(p3_violated, False, "The point lies "
                                                     "on the constraint "
                                                     "therefore the value of the funciton is "
                                                     "violated should be "
                                                     "False")

        def test_constraint(self):

            p1_out = self._constr.constraint(self._p1_inside)
            p2_out = self._constr.constraint(self._p2_outside)
            p3_out = self._constr.constraint(self._p3_on)

            self.logger.info("The value of the constraint for the point {:} "
                             "is {:}".format(self._p1_inside, p1_out))
            self.logger.info("The value of the constraint for the point {:} "
                             "is {:}".format(self._p2_outside, p2_out))
            self.logger.info("The value of the constraint for the point {:} "
                             "is {:}".format(self._p3_on, p3_out))

            self.assertLess(p1_out, 0, "The point lies "
                                                     "inside the constraint "
                                       "therefore the value of the constraint "
                                       "function should be negative")
            self.assertGreater(p2_out, 0, "The point lies "
                                                     "outside the constraint "
                                       "therefore the value of the constraint "
                                       "function should be positive")
            self.assertEqual(p3_out, 0, "The point lies "
                                                     "on the constraint "
                                       "therefore the value of the constraint "
                                       "function should be equal to zero")

    # TODO: READD
        def _test_gradient(self):
            """
            Test the gradient of the constraint function
            """
            # Compare the analytical grad with the numerical grad
            gradient = self._constr.gradient(self._p1_inside)
            num_gradient = CFunction(self._constr.constraint,
                                     self._constr.gradient).approx_fprime(
                self._p1_inside, 1e-8)
            error = (gradient - num_gradient).norm(order=1)
            self.logger.info("Analitic gradient {:}".format(str(gradient)))
            self.logger.info(
                "Numerical gradient {:}".format(str(num_gradient)))

            self.logger.info(
                "norm(grad - num_grad): %s", str(error))
            self.assertLess(error, 1e-3, "the gradient function of the "
                                         "constraint object does not work")

        def test_projection(self):

            pout = CArray([0,2])
            real_proj_pout = CArray([0,1])
            pout_proj = self._constr.projection(pout)

            error = (real_proj_pout - pout_proj).norm(order=2)
            self.logger.info(
                "norm(real projected point - projected point): %s", str(error))
            self.assertLess(error, 1e-3, "the projection function is not "
                                         "working correclty")

        def setUp(self):
            self._constr_creation()
            self._set_constr_name()

            self._create_test_points() # creates the point that we will use
            # to test the constraint functionalities
