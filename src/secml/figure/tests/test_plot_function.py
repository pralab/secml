from secml.testing import CUnitTest

from secml.figure import CFigure
from secml.ml.classifiers import CClassifierSVM
from secml.data.loader import CDLRandom
from secml.ml.features.normalization import CNormalizerMinMax


class TestCPlot(CUnitTest):
    """Unit test for TestCPlot."""

    def setUp(self):
        self.clf = CClassifierSVM()
        self.dataset = CDLRandom(n_features=2, n_redundant=0, n_informative=1,
                                 n_clusters_per_class=1).load()
        self.dataset.X = CNormalizerMinMax().fit_transform(self.dataset.X)
        self.clf.fit(self.dataset)

    def test_fun(self):
        """Test for CPlotFunction.plot_fun method."""
        fig = CFigure()
        fig.sp.plot_ds(self.dataset)

        fig.sp.plot_fun(self.clf.decision_function, y=1)
        fig.show()

    def test_fgrads(self):
        """Test for CPlotFunction.plot_fgrads method."""
        fig = CFigure()
        fig.sp.plot_ds(self.dataset)

        fig.sp.plot_fun(self.clf.decision_function, y=1)
        fig.sp.plot_fgrads(lambda x: self.clf.grad_f_x(x))
        fig.show()


if __name__ == '__main__':
    CUnitTest.main()
