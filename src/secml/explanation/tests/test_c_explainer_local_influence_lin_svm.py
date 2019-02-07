from secml.explanation.tests import CExplainerLocalInfluenceTestCases

from secml.ml.classifiers import CClassifierSVM


class TestCExplainerLocalInfluenceLinSVM(CExplainerLocalInfluenceTestCases):
    """Unit test for CExplainerLocalInfluenceSVM."""

    def _clf_creation(self):
        self._clf = CClassifierSVM()
        self._clf.store_dual_vars = True

    def test_explanation(self):
        self.logger.info("Explain the decisions of an SVM classifier and "
                         "test if they are reasonable")
        self._test_explanation()


if __name__ == '__main__':
    TestCExplainerLocalInfluenceLinSVM.main()
