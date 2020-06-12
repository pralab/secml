from secml.data.loader import CDataLoaderMNIST
from secml.ml.classifiers import CClassifierSVM
from secml.ml.classifiers.sklearn.c_classifier_svm_m import CClassifierSVMM
from secml.ml.kernels import CKernelRBF
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.ml.peval.metrics import CMetricAccuracy

random_state = 999

digits = (4, 6)  # tuple(range(0, 10))

n_tr = 500  # Number of training set samples
n_ts = 1000  # Number of test set samples

loader = CDataLoaderMNIST()
tr = loader.load('training', digits=digits, num_samples=n_tr)
ts = loader.load('testing', digits=digits, num_samples=n_ts)

# Normalize the features in `[0, 1]`
tr.X /= 255
ts.X /= 255

# Force storing of the dual space variables (alphas and support vectors)
# Will be used by the poisoning attack later
kernel = CKernelRBF(gamma=0.1)
C = 100
classifiers = [
    CClassifierMulticlassOVA(CClassifierSVM, kernel=kernel, C=100),
    CClassifierSVMM(kernel=kernel, C=100),
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

print(clf.alpha.shape)
print((grads[0] - grads[1]).norm())
