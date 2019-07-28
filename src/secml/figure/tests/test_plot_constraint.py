from secml.testing import CUnitTest

from secml.figure import CFigure
from secml.optim.constraints import CConstraint


class TestCPlotConstraint(CUnitTest):
    """Unit test for TestCPlot."""

    def setUp(self):
        self.constraints = [
            CConstraint.create("box", lb=0, ub=1),
            CConstraint.create("l1", center=0.5, radius=0.5),
            CConstraint.create("l2", center=0.5, radius=0.5)
        ]

    def test_constraint(self):
        """Test for CPlotFunction.plot_fun method."""
        fig = CFigure()
        for constraint in self.constraints:
            fig.sp.plot_constraint(constraint)
        fig.show()


if __name__ == '__main__':
    CUnitTest.main()
