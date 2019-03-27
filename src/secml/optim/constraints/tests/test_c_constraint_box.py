from secml.utils import CUnitTest
from secml.optim.constraints.tests.test_c_constraint import CConstraintTestCases

from secml.optim.constraints import CConstraintBox
from secml.core.constants import inf

class TestConstraintBox(CConstraintTestCases.TestCConstraint):
    """Test the L-inf distance constraint."""

    def _constr_creation(self):
        lb = -1
        ub = 1
        self._constr = CConstraintBox(lb=lb, ub=ub)

    def _set_constr_name(self):
        self._constr_name = 'L-inf'

    def _set_norm_order(self):
        """
        Set the distance on which the constrain is computed
        """
        self._norm_order = inf

if __name__ == '__main__':
    CUnitTest.main()
