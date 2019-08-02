from secml.testing import CUnitTest

from secml.figure import CFigure
from secml.ml.classifiers import CClassifierSVM
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.data.loader import CDLRandomBlobs
from secml.ml.features.normalization import CNormalizerMinMax


class TestCPlotClassifier(CUnitTest):
    """Unit test for CPlotClassifier."""

    def setUp(self):
        self.clf = CClassifierMulticlassOVA(
            classifier=CClassifierSVM, kernel='rbf')
        self.dataset = CDLRandomBlobs(
            random_state=3, n_features=2, centers=4).load()
        self.dataset.X = CNormalizerMinMax().fit_transform(self.dataset.X)
        self.clf.fit(self.dataset)

    def test_plot_decision_regions(self):
        """Test for `.plot_decision_regions` method."""
        fig = CFigure(width=10, height=5)

        fig.subplot(1, 2, 1)
        fig.sp.plot_ds(self.dataset)
        fig.sp.plot_decision_regions(
            self.clf, n_grid_points=200, plot_background=False)

        fig.subplot(1, 2, 2)
        fig.sp.plot_ds(self.dataset)
        fig.sp.plot_decision_regions(
            self.clf, n_grid_points=200)

        fig.show()


if __name__ == '__main__':
    CUnitTest.main()
