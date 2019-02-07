from secml.explanation.tests import CExplainerLocalInfluenceTestCases
from secml.ml.features.normalization import CNormalizerMinMax
from secml.ml.classifiers import CClassifierSVM


class TestCExplainerLocalInfluenceRbfSVM(CExplainerLocalInfluenceTestCases):
    """Unit test for CExplainerLocalInfluenceRBFSVM."""

    def _clf_creation(self):
        self._clf = CClassifierSVM(kernel='rbf', C=10)
        self._clf.kernel.gamma = 0.01
        self._clf.store_dual_vars = True

    def test_explanation(self):
        self.logger.info("Explain the decisions of an SVM classifier and "
                         "test if they are reasonable")
        self._test_explanation()

    def test_explanation_with_normalization(self):
        self.logger.info("Explain the decisions of an RBF-SVM classifier with "
                         "a normalizer inside and "
                         "test if they are reasonable")

        normalizer = CNormalizerMinMax(feature_range=(-10,10))
        normalizer.fit(self._tr.X)
        self._clf.preprocess = normalizer

        self._test_explanation()


if __name__ == '__main__':
    TestCExplainerLocalInfluenceRbfSVM.main()
