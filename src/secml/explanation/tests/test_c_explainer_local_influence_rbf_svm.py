from secml.explanation.tests import CExplainerLocalInfluenceTestCases
from secml.ml.classifiers import CClassifierSVM


class TestCExplainerLocalInfluenceRbfSVM(CExplainerLocalInfluenceTestCases):
    """Unit test for CExplainerLocalInfluenceRBFSVM."""

    def _clf_creation(self):
        self._clf = CClassifierSVM(kernel='rbf', C=10)
        self._clf.kernel.gamma = 0.01
        self._clf.store_dual_vars = True
        self._clf_idx = 'rbf-svm'
        vals = [1e-4, 1e-3, 1e-2, 1, 1e-2, 1e3, 1e4]
        self._param_values = {'C': vals, 'gamma': vals}

    def test_explanation(self):
        self._test_explanation_simple_clf()

    def test_explanation_with_normalization(self):
        self._test_explanation_with_normalization()

    def test_explanation_with_feat_nn_extraction(self):
        self._test_explanation_with_feat_nn_extraction()


if __name__ == '__main__':
    TestCExplainerLocalInfluenceRbfSVM.main()
