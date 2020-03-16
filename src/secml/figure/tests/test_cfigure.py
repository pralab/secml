from secml.testing import CUnitTest

from secml.array import CArray
from secml.data import CDataset
from secml.figure import CFigure
from secml.ml.classifiers import CClassifierSVM
from secml.ml.kernels import CKernelRBF


class TestCFigure(CUnitTest):
    """Unittest for CFigure."""
    
    def test_svm(self):

        self.X = CArray([[1, 2], [3, 4], [5, 6], [7, 8]])
        self.Y = CArray([[0], [1], [1], [0]]).ravel()
        self.dataset = CDataset(self.X, self.Y)

        self.classifier = CClassifierSVM(kernel=CKernelRBF())
        self.classifier.fit(self.dataset)

        self.x_min, self.x_max = (self.X[:, [0]].min() - 1,
                                  self.X[:, [0]].max() + 1)
        self.y_min, self.y_max = (self.X[:, [1]].min() - 1,
                                  self.X[:, [1]].max() + 1)

        self.fig = CFigure(height=7, width=10,
                           linewidth=5, fontsize=24, markersize=20)
        self.fig.sp.title("Svm Test")

        self.logger.info("Test plot dataset method...")

        self.fig.sp.plot_ds(self.dataset)

        self.logger.info("Test plot path method...")
        path = CArray([[1, 2], [1, 3], [1.5, 5]])
        self.fig.sp.plot_path(path)

        self.logger.info("Test plot function method...")
        bounds = [(self.x_min, self.x_max), (self.y_min, self.y_max)]
        self.fig.sp.plot_fun(self.classifier.decision_function,
                             plot_levels=False, grid_limits=bounds, y=1)

        self.fig.sp.xlim(self.x_min, self.x_max)
        self.fig.sp.ylim(self.y_min, self.y_max)

        self.fig.show()


if __name__ == '__main__':
    CUnitTest.main()
