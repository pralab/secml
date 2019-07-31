from secml.testing import CUnitTest
from secml.ml.classifiers.reject.tests import CClassifierRejectTestCases

from secml.data.loader import CDLRandomBlobs
from secml.ml.classifiers import CClassifierSGD
from secml.ml.classifiers.reject import CClassifierRejectThreshold
from secml.ml.classifiers.loss import *
from secml.ml.classifiers.regularizer import *
from secml.figure import CFigure


class TestCClassifierRejectThreshold(
    CClassifierRejectTestCases.TestCClassifierReject):
    """Unit test for Classifiers that reject based on a defined threshold."""

    def setUp(self):
        """Test for init and fit methods."""
        # generate synthetic data
        self.dataset = CDLRandomBlobs(n_features=2, n_samples=100, centers=2,
                                      cluster_std=2.0, random_state=0).load()

        self.logger.info("Testing classifier creation ")
        self.clf_norej = CClassifierSGD(regularizer=CRegularizerL2(),
                                        loss=CLossHinge(), random_state=0)

        self.clf = CClassifierRejectThreshold(self.clf_norej, threshold=0.6)
        self.clf.verbose = 2  # Enabling debug output for each classifier
        self.clf.fit(self.dataset)

    def test_draw(self):
        """ Compare the classifiers graphically"""
        self.logger.info("Testing classifiers graphically")

        fig = CFigure(width=10, markersize=8)
        # Plot dataset points

        # mark the rejected samples
        y = self.clf.predict(self.dataset.X)
        fig.sp.plot_ds(
            self.dataset[y == -1, :], colors=['k', 'k'], markersize=12)

        # plot the dataset
        fig.sp.plot_ds(self.dataset)

        # Plot objective function
        fig.sp.plot_fun(self.clf.decision_function,
                        grid_limits=self.dataset.get_bounds(),
                        levels=[0], y=1)
        fig.sp.title('Classifier with reject threshold')

        fig.show()


if __name__ == '__main__':
    CUnitTest.main()
