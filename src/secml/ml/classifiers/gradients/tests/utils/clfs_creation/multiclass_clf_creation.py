from secml.ml.features.normalization import CNormalizerMinMax
from secml.ml.classifiers import CClassifierSVM
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA


def multiclass_clf_creation(clf_idx, normalizer=False, dataset=None):
    if clf_idx == 'OVA':
        clf = CClassifierMulticlassOVA(CClassifierSVM)
    else:
        raise ValueError("classifier idx not managed!")

    if normalizer:
        normalizer = CNormalizerMinMax((-10, 10))
        clf.preprocess = normalizer

    return clf
