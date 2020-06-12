from secml.data.loader import CDataLoaderMNIST
from secml.ml.classifiers.sklearn.c_classifier_svm import CClassifierSVM as CClassifierSVMO
from secml.ml.classifiers import CClassifierSVM
from secml.ml.kernels import CKernelRBF
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.ml.peval.metrics import CMetricAccuracy

random_state = 999

digits = tuple(range(0, 10))

n_tr = 800  # Number of training set samples
n_ts = 1000  # Number of test set samples

loader = CDataLoaderMNIST()
tr = loader.load('training', digits=digits, num_samples=n_tr)
ts = loader.load('testing', digits=digits, num_samples=n_ts)

# Normalize the features in `[0, 1]`
tr.X /= 255
ts.X /= 255

# Force storing of the dual space variables (alphas and support vectors)
# Will be used by the poisoning attack later
svm_params = {
    'kernel': CKernelRBF(gamma=0.1),
    'C': 10,
    'class_weight': {0: 1, 1: 1},
    'store_dual_vars': None
}
classifiers = [
    CClassifierMulticlassOVA(CClassifierSVMO, **svm_params),
    CClassifierSVM(**svm_params),
]

grads = []
for clf in classifiers:
    # We can now fit the classifier
    print("Fit")
    clf.fit(tr.X, tr.Y)

    # Compute predictions on a test set
    print("Predict")
    y_pred, scores = clf.predict(ts.X, return_decision_function=True)

    # Evaluate the accuracy of the classifier
    metric = CMetricAccuracy()
    acc = metric.performance_score(y_true=ts.Y, y_pred=y_pred)

    print("Accuracy on test set: {:.2%}".format(acc))

    grads.append(clf.grad_f_x(ts.X[1, :], 1))

if clf.w is not None:
    print("w: ", clf.w.shape)

if clf.alpha is not None:
    print("alpha: ", clf.alpha.shape)

print('grad-diff: ', (grads[0] - grads[1]).norm())

clf.set('kernel.gamma', 0.01)