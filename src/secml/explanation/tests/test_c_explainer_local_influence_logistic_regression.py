from secml.explanation.tests import CExplainerLocalInfluenceTestCases
from secml.ml.classifiers import CClassifierLogistic


class TestCExplainerLocalInfluenceLogisticRegression(
    CExplainerLocalInfluenceTestCases):
    """Unit test for CExplainerLocalInfluenceLogisticRegression."""

    def _clf_creation(self):
        self._clf = CClassifierLogistic()
        self._clf_idx = 'logistic regression'
        self._param_values = {}

    def test_explanation(self):
        self._test_explanation_simple_clf()

    def test_explanation_with_feat_nn_extraction(self):
        self._test_explanation_with_feat_nn_extraction()


if __name__ == '__main__':
    TestCExplainerLocalInfluenceLogisticRegression.main()
