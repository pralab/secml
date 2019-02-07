from secml.explanation.tests import CExplainerLocalInfluenceTestCases
from secml.ml.features.normalization import CNormalizerMinMax
from secml.ml.classifiers import CClassifierRidge


class TestCExplainerLocalInfluenceRidge(CExplainerLocalInfluenceTestCases):
    """Unit test for CExplainerLocalInfluenceRidge."""

    def _clf_creation(self):
        self._clf = CClassifierRidge()

    def test_explanation(self):
        self.logger.info("Explain the decisions of a Ridge classifier and "
                         "test if they are reasonable")
        self._test_explanation()

    def test_explanation_with_normalization(self):
        self.logger.info("Explain the decisions of a Ridge classifier with "
                         "a normalizer inside and "
                         "test if they are reasonable")

        normalizer = CNormalizerMinMax(feature_range=(-10,10))
        normalizer.fit(self._tr.X)
        self._clf.preprocess = normalizer

        self._test_explanation()


if __name__ == '__main__':
    TestCExplainerLocalInfluenceRidge.main()
