from secml.testing import CUnitTest
from secml.ml.classifiers.reject.tests import CClassifierRejectTestCases

from secml.data.loader import CDLRandomBlobs
from secml.ml.classifiers import CClassifierSGD
from secml.ml.classifiers.reject import CClassifierRejectThreshold
from secml.ml.classifiers.loss import *
from secml.ml.classifiers.regularizer import *


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


if __name__ == '__main__':
    CUnitTest.main()
