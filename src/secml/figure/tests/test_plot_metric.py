from secml.testing import CUnitTest

from secml.figure import CFigure
from secml.array import CArray


class TestCPlotClassifier(CUnitTest):
    """Unittests for CPlotMetric."""

    def test_confusion_matrix(self):
        """Test for `CPlot.plot_confusion_matrix()` method."""
        y_true = CArray([2, 0, 2, 2, 0, 1])
        y_pred = CArray([0, 0, 2, 2, 0, 2])
        fig = CFigure()
        fig.sp.plot_confusion_matrix(
            y_true, y_pred, labels=['one', 'two', 'three'],
            colorbar=True, normalize=False)
        fig.show()


if __name__ == '__main__':
    CUnitTest.main()
