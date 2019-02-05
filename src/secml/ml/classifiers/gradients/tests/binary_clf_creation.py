from secml.ml.features.normalization import CNormalizerMinMax
from secml.ml.classifiers.loss import CLossHinge
from secml.ml.classifiers.regularizer import CRegularizerL2
from secml.ml.classifiers import CClassifierSVM, CClassifierKDE, \
    CClassifierSGD, CClassifierMCSLinear, CClassifierLogistic, CClassifierRidge

def clf_creation(clf_idx, normalizer=False):

    if clf_idx == 'lin-svm':
        clf = CClassifierSVM()
    elif clf_idx == 'rbf-svm':
        CClassifierSVM(kernel='rbf')
    elif clf_idx == 'logistic':
        clf = CClassifierLogistic()
    elif clf_idx == 'ridge':
        clf = CClassifierRidge()
    elif clf_idx == 'lin-mcs':
        clf = CClassifierMCSLinear(CClassifierSVM(),
                                          num_classifiers=3,
                                          max_features=0.5,
                                          max_samples=0.5,
                                          random_state=0)
    elif clf_idx == 'kde':
        clf = CClassifierKDE()
    elif clf_idx == 'sgd-lin':
        clf = CClassifierSGD(CLossHinge(),CRegularizerL2())
    elif clf_idx == 'sgd-rbf':
        CClassifierSGD(CLossHinge(), CRegularizerL2(),kernel='rbf')
    else:
        raise ValueError("classifier idx not managed!")

    if normalizer:
        normalizer = CNormalizerMinMax((-10, 10))
        clf.preprocess = normalizer

    return clf