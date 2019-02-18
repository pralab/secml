from secml.explanation.tests import CExplainerLocalInfluenceTestCases
from secml.ml.classifiers import CClassifierRidge


class TestCExplainerLocalInfluenceRidge(CExplainerLocalInfluenceTestCases):
    """Unit test for CExplainerLocalInfluenceRidge."""

    def _clf_creation(self):
        self._clf = CClassifierRidge()
        self._clf_idx = 'Ridge'
        self._param_values = {}

    def test_explanation(self):
        self._test_explanation_simple_clf()

    # def test_explanation_with_normalization(self):
    #     self._test_explanation_with_normalization()

    def test_explanation_with_feat_nn_extraction(self):
        self._test_explanation_with_feat_nn_extraction()


if __name__ == '__main__':
    TestCExplainerLocalInfluenceRidge.main()
