from secml.testing import CUnitTest

from secml.figure import CFigure
from secml.ml.classifiers import CClassifierSVM
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.data.loader import CDLRandomBlobs
from secml.ml.features.normalization import CNormalizerMinMax

from matplotlib.colors import ListedColormap


class TestCPlotDecisionFunction(CUnitTest):
    """Unit test for TestCPlot."""

    def setUp(self):
        self.clf = CClassifierMulticlassOVA(
            classifier=CClassifierSVM, kernel='rbf')
        self.dataset = CDLRandomBlobs(
            random_state=3, n_features=2, centers=4).load()
        self.dataset.X = CNormalizerMinMax().fit_transform(self.dataset.X)
        self.clf.fit(self.dataset)

    def test_exception(self):
        fig = CFigure()
        self.assertRaises(TypeError, fig.sp.plot_decision_regions, 'aaaa')

    def test_decision_function(self):
        """Test for CPlotFunction.plot_fobj method."""

        # custom colormap (better than how we discretize jet)
        colors = ('red', 'blue', 'lightgreen', 'black', 'gray', 'cyan')
        cmap = ListedColormap(colors[:self.clf.n_classes])

        fig = CFigure(width=10, height=5)

        fig.subplot(1, 2, 1)
        fig.sp.plot_decision_regions(self.clf, n_grid_points=200, cmap=cmap,
                                     plot_background=False)
        fig.sp.plot_ds(self.dataset, cmap=cmap)
        fig.sp.grid(grid_on=False)

        fig.subplot(1, 2, 2)
        fig.sp.plot_decision_regions(self.clf, n_grid_points=200, cmap=cmap)
        fig.sp.plot_ds(self.dataset, cmap=cmap)
        fig.sp.grid(grid_on=False)

        fig.savefig('test_plot_decision_function.pdf')


if __name__ == '__main__':
    CUnitTest.main()
