import unittest
from secml.utils import CUnitTest
from secml.figure import CFigure
from secml.array import CArray


class TestCPlotConfMatr(CUnitTest):
    """Unit test for TestCPlot."""

    def setUp(self):
        self.conf_matr = CArray([[2, 3, 4], [10, 7, 9], [3, 2, 1]])

    def test_conf_matr(self):
        """Test for CPlotFunction.plot_fobj method."""
        fig = CFigure()
        fig.switch_sptype(sp_type='conf-matrix')
        fig.sp.plot_confusion_matrix(self.conf_matr, ['one', 'two', 'three'])

        fig.show()


if __name__ == '__main__':
    unittest.main()
