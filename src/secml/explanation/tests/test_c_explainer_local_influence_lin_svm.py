from secml.explanation.tests import CExplainerLocalInfluenceTestCases

from secml.ml.classifiers import CClassifierSVM


class TestCExplainerLocalInfluenceLinSVM(CExplainerLocalInfluenceTestCases):
    """Unit test for CExplainerLocalInfluenceSVM."""

    def _clf_creation(self):
        self._clf = CClassifierSVM()
        self._clf.store_dual_vars = True
        self._clf_idx = 'lin-svm'

    def test_explanation(self):
        self._test_explanation_simple_clf()

    # def test_explanation_with_normalization(self):
    #     self._test_explanation_with_normalization()

    def test_explanation_with_feat_nn_extraction(self):
        self._test_explanation_with_feat_nn_extraction()
        self._test_explanation()


if __name__ == '__main__':
    TestCExplainerLocalInfluenceLinSVM.main()
