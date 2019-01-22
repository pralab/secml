from secml.data.loader import CDLRandom
from secml.ml.classifiers.loss import CLossHinge
from secml.ml.classifiers.regularizer import CRegularizerL2
from secml.ml.classifiers import CClassifierSVM, CClassifierKDE, CClassifierSGD
from secml.utils import CUnitTest
from test_c_classifier_gradient import CClassifierGradientTestCases


class TestCClassifierGradientBinary(
    CClassifierGradientTestCases.TestCClassifierGradient):
    """Test of binary classifiers gradients."""

    def _dataset_creation(self):
        # generate synthetic data
        self.dataset = CDLRandom(n_features=2, n_redundant=0, n_informative=2,
                                 n_clusters_per_class=1, random_state=1).load()

        self.dataset_sparse = self.dataset.tosparse()

    def _clfs_creation(self):

        self.clfs = [CClassifierSVM(), CClassifierSVM(kernel='rbf'),
                     CClassifierKDE(), CClassifierSGD(CLossHinge(),
                      CRegularizerL2()),
                     CClassifierSGD(CLossHinge(),CRegularizerL2(),
                      kernel='rbf')]

        self.clf_ids = ['lin-SVM', 'rbf-SVM', 'lin-SGD', 'rbf-SGD','KDE']

        for clf in self.clfs:  # Enabling debug output for each classifier
            clf.verbose = 2


if __name__ == '__main__':
    CUnitTest.main()
