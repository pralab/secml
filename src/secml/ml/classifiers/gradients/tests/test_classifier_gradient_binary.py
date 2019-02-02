from secml.utils import CUnitTest

from test_c_classifier_gradient import CClassifierGradientTestCases
from secml.data.loader import CDLRandom
from secml.ml.classifiers.loss import CLossHinge
from secml.ml.classifiers.regularizer import CRegularizerL2
from secml.ml.classifiers import CClassifierSVM, CClassifierKDE, \
    CClassifierSGD, CClassifierMCSLinear, CClassifierLogistic


class TestCClassifierGradientBinary(
    CClassifierGradientTestCases.TestCClassifierGradient):
    """Compare the analytical and the numerical gradient of binary
    classifiers."""

    def _dataset_creation(self):
        # generate synthetic data
        self.dataset = CDLRandom(n_features=2, n_redundant=0, n_informative=2,
                                 n_clusters_per_class=1, random_state=1).load()
        self.dataset_sparse = self.dataset.tosparse()

    def _set_tested_classes(self):
        self.classes = self.dataset.classes

    def _clfs_creation(self):
        self.clfs = [CClassifierSVM(), CClassifierSVM(kernel='rbf'),
                     CClassifierLogistic(),
                     CClassifierMCSLinear(CClassifierSVM(),
                                          num_classifiers=3,
                                          max_features=0.5,
                                          max_samples=0.5,
                                          random_state=0),
                     CClassifierKDE(), CClassifierSGD(CLossHinge(),
                                                      CRegularizerL2()),
                     CClassifierSGD(CLossHinge(), CRegularizerL2(),
                                    kernel='rbf')]

        self.clf_ids = ['lin-SVM', 'rbf-SVM', 'logistic']

        for clf in self.clfs:  # Enabling debug output for each classifier
            clf.verbose = 2


if __name__ == '__main__':
    CUnitTest.main()
