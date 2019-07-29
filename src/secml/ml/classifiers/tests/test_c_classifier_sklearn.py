from secml.ml.classifiers.tests import CClassifierTestCases

from secml.data.loader import CDLIris
from secml.ml.classifiers import CClassifierSkLearn
from secml.array import CArray

from sklearn.svm import SVC


class TestCClassifierDecisionTree(CClassifierTestCases):
    """Unit test for CDecisionTree."""

    def setUp(self):
        self.dataset = CDLIris().load()
        self.sklearn_model = SVC()
        self.clf = CClassifierSkLearn(sklearn_model=self.sklearn_model)

    def test_fun(self):
        """Test for decision_function() and predict() methods."""

        # create a fake private decision_function to run tests
        def _decision_function(x, y):
            x = x.atleast_2d()
            scores = CArray(self.sklearn_model.decision_function(x.get_data()))
            return scores[:, y].ravel()

        self.clf._decision_function = _decision_function

        # execute tests
        self._test_fun(self.clf, self.dataset.todense())
        self._test_fun(self.clf, self.dataset.tosparse())

    def test_preprocess(self):
        """Test classifier with preprocessors inside."""
        # All linear transformations
        self._test_preprocess(self.dataset, self.clf,
                              ['min-max', 'mean-std'],
                              [{'feature_range': (-1, 1)}, {}])

        # Mixed linear/nonlinear transformations
        self._test_preprocess(self.dataset, self.clf,
                              ['pca', 'unit-norm'], [{}, {}])


if __name__ == '__main__':
    CClassifierTestCases.main()
