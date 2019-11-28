from secml.ml.classifiers.tests import CClassifierTestCases

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

from secml.data.loader import CDLRandom
from secml.ml.classifiers import CClassifierSkLearn
from secml.array import CArray
from secml.data import CDataset
from secml.ml.features import CPreProcess


class TestCClassifierSkLearn(CClassifierTestCases):
    """Unit test for SkLearn classifiers."""

    def setUp(self):

        # QuadraticDiscriminantAnalysis will raise a warning
        self.logger.filterwarnings(
            "ignore", message="Variables are collinear", category=UserWarning)

        multiclass = True

        n_classes = 3 if multiclass is True else 2
        self.dataset = CDLRandom(
            n_features=25, n_redundant=10, n_informative=5,
            n_classes=n_classes, n_samples=25,
            n_clusters_per_class=2, random_state=0).load()

        self.skclfs = [
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=0.025,
                random_state=0, decision_function_shape='ovr'),
            SVC(kernel="rbf", gamma=2, C=1, random_state=0),
            DecisionTreeClassifier(max_depth=5, random_state=0),
            RandomForestClassifier(max_depth=5, n_estimators=5,
                                   random_state=0),
            MLPClassifier(alpha=1, max_iter=1000, random_state=0),
            AdaBoostClassifier(random_state=0),
            OneVsRestClassifier(SVC(kernel='linear')),
            # These clf below only work on dense data!
            GaussianProcessClassifier(1.0 * RBF(1.0)),
            GaussianNB(),
            QuadraticDiscriminantAnalysis()
        ]

        self.classifiers = []
        for model in self.skclfs:
            self.classifiers.append(CClassifierSkLearn(sklearn_model=model))

    def test_fun(self):
        """Test for decision_function() and predict() methods."""

        for i, clf in enumerate(self.classifiers):

            self.logger.info("Classifier:\n - " + str(clf._sklearn_model))

            # create a fake private _decision_function to run tests
            # but this is basically the same in CClassifierSkLearn
            # - we need to think better tests!
            def _decision_function(x, y=None):
                x = x.atleast_2d()
                try:
                    scores = CArray(
                        self.skclfs[i].decision_function(x.get_data()))
                    probs = False
                except AttributeError:
                    scores = CArray(self.skclfs[i].predict_proba(x.get_data()))
                    probs = True

                # two-class classifiers outputting only scores for class 1
                if len(scores.shape) == 1:  # duplicate column for class 0
                    outputs = CArray.ones(shape=(x.shape[0], clf.n_classes))
                    scores = scores.T
                    outputs[:, 1] = scores
                    outputs[:, 0] = -scores if probs is False else 1 - scores
                    scores = outputs
                scores.atleast_2d()
                if y is not None:
                    return scores[:, y].ravel()
                else:
                    return scores

            clf._decision_function = _decision_function

            # execute tests
            self._test_fun(clf, self.dataset.todense())
            try:
                self._test_fun(clf, self.dataset.tosparse())
            except TypeError:
                self.logger.info(
                    "This sklearn model does not support sparse data!")

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

    def test_pretrained(self):
        """Test wrapping of pretrained models."""
        from sklearn import datasets, svm

        iris = datasets.load_iris()
        X = iris.data
        y = iris.target

        clf = svm.SVC(kernel='linear')

        from secml.core.exceptions import NotFittedError
        with self.assertRaises(NotFittedError):
            secmlclf = CClassifierSkLearn(clf)
            secmlclf.predict(CArray(X))

        clf.fit(X, y)
        
        y_pred = clf.predict(X)

        clf = svm.SVC(kernel='linear')
        secmlclf = CClassifierSkLearn(clf)
        secmlclf.fit(CDataset(X, y))

        y_pred_secml = secmlclf.predict(CArray(X))

        self.logger.info(
            "Predicted labels by pretrained model:\n{:}".format(y_pred))
        self.logger.info(
            "Predicted labels by our fit:\n{:}".format(y_pred_secml))

        self.assert_array_equal(y_pred, y_pred_secml)

    def test_set_get_state(self):
        """Test for set_state and get_state."""

        pre = CPreProcess.create_chain(['pca', 'mean-std'], [{}, {}])
        clf = CClassifierSkLearn(
            sklearn_model=SVC(kernel="rbf", gamma=2, C=1, random_state=0),
            preprocess=pre)

        clf.fit(self.dataset)
        pred_y = clf.predict(self.dataset.X)
        self.logger.info(
            "Predictions before restoring state:\n{:}".format(pred_y))

        state = clf.get_state()
        self.logger.info("State of multiclass:\n{:}".format(state))

        # Generate a temp file to test
        import tempfile
        from secml.utils import fm
        tempdir = tempfile.gettempdir()
        tempfile = fm.join(tempdir, 'secml_testgetsetstate')

        # Test save state to disk
        tempfile = clf.save_state(tempfile)

        # Create an entirely new clf
        pre_post = CPreProcess.create_chain(['pca', 'mean-std'], [{}, {}])
        clf_post = CClassifierSkLearn(
            sklearn_model=SVC(kernel="rbf", gamma=2, C=1, random_state=0),
            preprocess=pre_post)

        # Restore state from disk
        clf_post.load_state(tempfile)

        pred_y_post = clf_post.predict(self.dataset.X)
        self.logger.info(
            "Predictions after restoring state:\n{:}".format(pred_y_post))

        self.assert_array_equal(pred_y, pred_y_post)


if __name__ == '__main__':
    CClassifierTestCases.main()
