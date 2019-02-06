from secml.explanation.tests import CExplainerLocalInfluenceTestCases

from secml.ml.classifiers import CClassifierSVM


class TestCExplainerLocalInfluenceSVM(CExplainerLocalInfluenceTestCases):
    """Unit test for CExplainerLocalInfluenceSVM."""

    def _clf_creation(self):
        self._clf = CClassifierSVM()
        self._clf.store_dual_vars = True
        self._clf.fit(self._tr)

    def test_explanation(self):

        self.logger.info(self.influences.shape)

if __name__ == '__main__':
    TestCExplainerLocalInfluenceSVM.main()
