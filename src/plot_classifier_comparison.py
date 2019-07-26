import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
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

from secml.array import CArray
from secml.data.loader import CDLRandomMoons, CDLRandomBlobs
from secml.ml.features.normalization import CNormalizerMinMax
from secml.figure import CFigure

from matplotlib.colors import ListedColormap

from secml.ml.classifiers import CClassifierSkLearn


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
fig = CFigure(fontsize=7, width=10)
fig.subplot(1, 3, 1)
fig.sp.plot_decision_function(svm, n_grid_points=500, cmap=cmap)
fig.sp.plot_ds(data, cmap=cmap)
fig.sp.title('SecML OVA SVM')
fig.subplot(1, 3, 2)
clf = CClassifierSkLearn(OneVsRestClassifier(classifiers[1]))
clf.fit(data)
fig.sp.plot_decision_function(clf, n_grid_points=500, cmap=cmap)
fig.sp.plot_ds(data, cmap=cmap)
fig.sp.title('SkLearn OVA SVM')
fig.subplot(1, 3, 3)
clf = CClassifierSkLearn(classifiers[1])
clf.fit(data)
fig.sp.plot_decision_function(clf, n_grid_points=500, cmap=cmap)
fig.sp.plot_ds(data, cmap=cmap)
fig.sp.title('SkLearn OVO SVM')

fig.show()
