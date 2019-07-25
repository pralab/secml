from secml.testing import CUnitTest

from secml.figure import CFigure
from secml.array import CArray
from secml.core import constants


class TestCPlot(CUnitTest):
    """Unit test for TestCPlot."""

    def test_quiver(self):
        """Test for `CPlot.quiver()` method."""

        # gradient values creation
        xv = CArray.arange(0, 2 * constants.pi, .2)
        yv = CArray.arange(0, 2 * constants.pi, .2)

        X, Y = CArray.meshgrid((xv, yv))
        U = CArray.cos(X)
        V = CArray.sin(Y)

        plot = CFigure()
        plot.sp.title('Gradient arrow')

        plot.sp.quiver(U, V)

        plot.show()


if __name__ == '__main__':
    CUnitTest.main()
