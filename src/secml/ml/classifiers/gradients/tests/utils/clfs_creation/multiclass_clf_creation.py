from secml.ml.features.normalization import CNormalizerMinMax
from secml.ml.classifiers import CClassifierSVM
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.pytorch.classifiers import CClassifierPyTorchMLP


def multiclass_clf_creation(clf_idx, normalizer=False, dataset=None):
    if clf_idx == 'OVA':
        clf = CClassifierMulticlassOVA(CClassifierSVM)
    elif clf_idx == 'pytorch_nn':
        clf = CClassifierPyTorchMLP(dataset.num_features, hidden_dims=(
            3,), output_dims=dataset.num_classes, weight_decay=0, epochs=10,
                                    learning_rate=1e-1, momentum=0,
                                    random_state=0)
    else:
        raise ValueError("classifier idx not managed!")

    if normalizer:
        normalizer = CNormalizerMinMax((-10, 10))
        clf.preprocess = normalizer

    return clf
