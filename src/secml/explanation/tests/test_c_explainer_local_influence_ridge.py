from secml.explanation.tests import CExplainerLocalInfluenceTestCases

from secml.ml.classifiers import CClassifierRidge


class TestCExplainerLocalInfluenceRidge(CExplainerLocalInfluenceTestCases):
    """Unit test for CExplainerLocalInfluenceSVM."""

    def _clf_creation(self):
        self._clf = CClassifierRidge()

    def test_explanation(self):
        self.logger.info("Explain the decisions of a Ridge classifier and "
                         "test if they are reasonable")
        self._test_explanation()


if __name__ == '__main__':
    TestCExplainerLocalInfluenceRidge.main()
