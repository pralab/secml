from .c_classifier import CClassifier
from .c_classifier_linear import CClassifierLinearMixin
from .c_classifier_dnn import CClassifierDNN

from .sklearn.c_classifier_sklearn import CClassifierSkLearn
from .sklearn.c_classifier_decision_tree import CClassifierDecisionTree
from .sklearn.c_classifier_knn import CClassifierKNN
from .sklearn.c_classifier_logistic import CClassifierLogistic
from .sklearn.c_classifier_nearest_centroid import CClassifierNearestCentroid
from .sklearn.c_classifier_random_forest import CClassifierRandomForest
from .sklearn.c_classifier_ridge import CClassifierRidge
from .sklearn.c_classifier_sgd import CClassifierSGD
from .sklearn.c_classifier_svm import CClassifierSVM

try:
    import torch
except ImportError:
    pass  # pytorch is an extra component
else:
    from .pytorch.c_classifier_pytorch import CClassifierPyTorch
