from secml.utils import CUnitTest
from test_c_constraint import CConstraintTestCases

from secml.optim.constraints import CConstraintL2


class TestConstraintL2(CConstraintTestCases.TestCConstraint):
    """Test the L2 distance constraint."""

    def _constr_creation(self):
        center = 0
        radius = 1
        self._constr = CConstraintL2(center=center, radius=radius)

    def _set_constr_name(self):
        self._constr_name = 'L2'

    def _set_norm_order(self):
        """
        Set the distance on which the constrain is computed
        """
        self._norm_order = 2

if __name__ == '__main__':
    CUnitTest.main()
