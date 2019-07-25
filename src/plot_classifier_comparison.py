import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from secml.array import CArray
from secml.data.loader import CDLRandomMoons, CDLRandomBlobs
from secml.ml.features.normalization import CNormalizerMinMax
from secml.figure import CFigure
from secml.ml.classifiers import CClassifier

from matplotlib.colors import ListedColormap


class CClassifierSkLearn(CClassifier):
    __class_type = 'sklearn-clf'

    def __init__(self, sklearn_model, preprocess=None):
        CClassifier.__init__(self, preprocess=preprocess)
        self._sklearn_model = sklearn_model

    def _fit(self, dataset):
        # TODO: handle sparse data...
        self._sklearn_model.fit(dataset.X.tondarray(), dataset.Y.tondarray())

    def predict(self, x, return_decision_function=False, n_jobs=1):
        """Perform classification of each pattern in x.

        If a preprocess has been specified,
         input is normalized before classification.

        Parameters
        ----------
        return_decision_function
        x : CArray
            Array with new patterns to classify, 2-Dimensional of shape
            (n_patterns, n_features).
        return_decision_function : bool, optional
            Whether to return the decision_function value along
            with predictions. Default False.
        n_jobs : int, optional
            Number of parallel workers to use for classification.
            Default 1. Cannot be higher than processor's number of cores.

        Returns
        -------
        labels : CArray
            Flat dense array of shape (n_patterns,) with the label assigned
             to each test pattern. The classification label is the label of
             the class associated with the highest score.
        scores : CArray, optional
            Array of shape (n_patterns, n_classes) with classification
             score of each test pattern with respect to each training class.
            Will be returned only if `return_decision_function` is True.

        Warnings
        --------
        This method implements a generic formulation where the
         classification score is computed separately for training class.
         It's convenient to override this when the score can be computed
         for one of the classes only, e.g. for binary classifiers the score
         for the positive/negative class is commonly the negative of the
         score of the other class.

        """
        x = x.atleast_2d()  # Ensuring input is 2-D

        #
        if hasattr(self._sklearn_model, "decision_function"):
            scores = self._sklearn_model.decision_function(x.tondarray())
            probs = False
        else:
            scores = self._sklearn_model.predict_proba(x.tondarray())
            probs = True
        scores = CArray(scores)

        # two-class classifiers outputting only scores for class 1
        if len(scores.shape) == 1:  # duplicate column for class 0
            outputs = CArray.ones(shape=(x.shape[0], self.n_classes))
            outputs[:, 1] = scores.T
            outputs[:, 0] = -scores.T if probs is False else 1 - scores.T
            scores = outputs

        if scores.shape[1] != self.n_classes:  # this happens in one-vs-one
            raise ValueError(
                "Number of columns is not equal to number of classes!")

        # The classification label is the label of the class
        # associated with the highest score
        labels = scores.argmax(axis=1).ravel()

        return (labels, scores) if return_decision_function is True else labels

    def _decision_function(self, x, y):
        raise NotImplementedError("Not implemented/required!")


names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, decision_function_shape='ovr'),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

n_classes = 4

# data = CDLRandomMoons(noise=0.1).load()
data = CDLRandomBlobs(random_state=3, n_features=2, centers=n_classes).load()
scaler = CNormalizerMinMax()
data.X = scaler.fit_transform(data.X, data.Y)

colors = ('red', 'blue', 'lightgreen', 'black', 'gray', 'cyan')
cmap = ListedColormap(colors[:n_classes])

fig = CFigure(width=10, height=8, fontsize=7)
for i in range(len(classifiers)):
    print("Classifier: " + names[i])
    sklearn_model = classifiers[i]
    clf = CClassifierSkLearn(sklearn_model)
    clf.fit(data)
    error = (clf.predict(data.X) != data.Y).mean()
    print("  - training error: " + str(round(error * 100, 1)) + "%")
    fig.subplot(2, 5, i + 1)
    fig.sp.plot_decision_function(clf, n_grid_points=500, cmap=cmap)
    fig.sp.plot_ds(data, cmap=cmap)
    fig.sp.title(names[i] + " - tr-err.: " + str(round(error * 100, 1)) + "%")

fig.show()

from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.ml.classifiers import CClassifierSVM

svm = CClassifierMulticlassOVA(CClassifierSVM, kernel='linear', C=0.025)
svm.fit(data)
fig = CFigure(fontsize=7)
fig.subplot(1, 2, 1)
fig.sp.plot_decision_function(svm, n_grid_points=500, cmap=cmap)
fig.sp.plot_ds(data, cmap=cmap)
fig.sp.title('SecML linear SVM')
fig.subplot(1, 2, 2)
clf = CClassifierSkLearn(classifiers[1])
clf.fit(data)
fig.sp.plot_decision_function(clf, n_grid_points=500, cmap=cmap)
fig.sp.plot_ds(data, cmap=cmap)
fig.sp.title('SkLearn linear SVM')

fig.show()
