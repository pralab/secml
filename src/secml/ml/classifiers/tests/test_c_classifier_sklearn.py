from secml.ml.classifiers.tests import CClassifierTestCases

from secml.data.loader import CDLIris
from secml.ml.classifiers import CClassifierSkLearn
from secml.array import CArray

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


class TestCClassifierDecisionTree(CClassifierTestCases):
    """Unit test for CDecisionTree."""

    def setUp(self):
        self.dataset = CDLIris().load()

        self.skclfs = [
            # KNeighborsClassifier(3),
            SVC(kernel="linear", C=0.025, decision_function_shape='ovr'),
            SVC(kernel="rbf", gamma=2, C=1),
            # DecisionTreeClassifier(max_depth=5),
            # RandomForestClassifier(max_depth=5, n_estimators=10,
            #                       max_features=1),
            # MLPClassifier(alpha=1, max_iter=1000),
            # AdaBoostClassifier(),
            # These clf below only work on dense data!
            # GaussianProcessClassifier(1.0 * RBF(1.0)),
            # GaussianNB(),
            # QuadraticDiscriminantAnalysis()
        ]

        self.classifiers = []
        for model in self.skclfs:
            self.classifiers.append(CClassifierSkLearn(sklearn_model=model))

    def test_fun(self):
        """Test for decision_function() and predict() methods."""

        for i, clf in enumerate(self.classifiers):

            # create a fake private decision_function to run tests
            def _decision_function(x, y):
                x = x.atleast_2d()
                scores = CArray(self.skclfs[i].decision_function(x.get_data()))
                return scores[:, y].ravel()
            clf._decision_function = _decision_function

            # execute tests
            self._test_fun(clf, self.dataset.todense())
            self._test_fun(clf, self.dataset.tosparse())

    def test_preprocess(self):
        """Test classifier with preprocessors inside."""
        # All linear transformations
        for clf in self.classifiers:
            self._test_preprocess(self.dataset, clf,
                                  ['min-max', 'mean-std'],
                                  [{'feature_range': (-1, 1)}, {}])

            # Mixed linear/nonlinear transformations
            self._test_preprocess(self.dataset, clf,
                                  ['pca', 'unit-norm'], [{}, {}])


if __name__ == '__main__':
    CClassifierTestCases.main()
