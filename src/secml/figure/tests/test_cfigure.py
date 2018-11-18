"""
Created on 27/apr/2015
Class to test CFigure
@author: Davide Maiorca
@author: Ambra Demontis

"""
from secml.utils import CUnitTest

from secml.array import CArray
from secml.data import CDataset
from secml.figure import CFigure
from secml.ml.classifiers import CClassifierSVM
from secml.core import constants


class TestCFigure(CUnitTest):
    """Unit test for CFigure."""

    @classmethod
    def setUpClass(cls):

        super(TestCFigure, cls).setUpClass()

        cls.X = CArray([[1, 2], [3, 4], [5, 6], [7, 8]])
        cls.Y = CArray([[0], [1], [1], [0]]).ravel()
        cls.dataset = CDataset(cls.X, cls.Y)

        cls.x_min, cls.x_max = (cls.X[:, [0]].min() - 1,
                                cls.X[:, [0]].max() + 1)
        cls.y_min, cls.y_max = (cls.X[:, [1]].min() - 1,
                                cls.X[:, [1]].max() + 1)

        cls.fig = CFigure(height=7, width=10,
                          linewidth=5, fontsize=24, markersize=20)
        cls.fig.sp.title("Svm Test")
        cls.fig.subplot(sp_type='ds')
        cls.fig.sp.xlim(cls.x_min, cls.x_max)
        cls.fig.sp.ylim(cls.y_min, cls.y_max)

    def setUp(self):
        """Set up method for tests."""
        self.logger.info("Starting Test...")
        self.logger.info("." * 50)
        from secml.ml.kernel import CKernelRBF
        self.classifier = CClassifierSVM(kernel=CKernelRBF())
        self.classifier.train(self.dataset)

    def test_plot_dataset_points(self):
        """Test plot dataset points method."""
        self.logger.info("Test plot dataset method...")
        # A good box bound can be, for each axis: (min()-1;max()+1)
        try:
            self.fig.switch_sptype('ds')
            self.fig.sp.plot_ds(self.dataset)
        except Exception as e:
            self.logger.info("Error: {:}".format(e))
            self.assertTrue(False, "Cannot plot dataset points!")

    def test_plot_path(self):
        """Test plot path method."""
        self.logger.info("Test plot path method...")
        path = CArray([[1, 2], [1, 3], [1.5, 5]])
        try:
            self.fig.sp.plot_path(path)
        except Exception as e:
            self.logger.info("Error: {:}".format(e))
            self.assertTrue(False, "Cannot plot path!")

    def test_plot_fobj(self):
        """Test plot fobj method."""
        self.logger.info("Test plot function method...")
        bounds = [(self.x_min, self.x_max), (self.y_min, self.y_max)]
        try:
            self.fig.switch_sptype('function')
            self.fig.sp.plot_fobj(self.classifier.discriminant_function,
                                  plot_levels=False, grid_limits=bounds)
        except Exception as e:
            self.logger.info("Error: {:}".format(e))
            self.assertTrue(False, "Cannot plot function!")

    def test_quiver(self):
        """Plot gradient arrows."""

        # gradient values creation
        xv = CArray.arange(0, 2 * constants.pi, .2)
        yv = CArray.arange(0, 2 * constants.pi, .2)

        X, Y = CArray.meshgrid((xv, yv))
        U = CArray.cos(X)
        V = CArray.sin(Y)

        plot = CFigure()
        plot.sp.title('Gradient arrow')

        plot.sp.quiver(U, V, X, Y)

        plot.show()

    @classmethod
    def tearDownClass(cls):
        cls.fig.show()


if __name__ == '__main__':
    CUnitTest.main()
