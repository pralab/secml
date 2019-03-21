from secml.utils import CUnitTest

from secml.ml.classifiers.gradients.tests import \
    CClassifierGradientTestCases, binary_clf_creation
from secml.data.loader import CDLRandom


class TestCClassifierGradientBinary(
    CClassifierGradientTestCases.TestCClassifierGradient):
    """Compare the analytical and the numerical gradient of binary
    classifiers."""

    @property
    def clf_list(self):
        return ['ridge', 'logistic', 'lin-svm', 'rbf-svm', 'lin-mcs']
        # fixme: read those classifier as soon as we will have fix their
        #  gradients
        #'sgd-lin','sgd-rbf','kde',

    @property
    def clf_creation_function(self):
        return binary_clf_creation

    def _dataset_creation(self):
        # generate synthetic data
        self.dataset = CDLRandom(n_features=2, n_redundant=0, n_informative=2,
                                 n_clusters_per_class=1,
                                 random_state=self.seed).load()
        self.dataset_sparse = self.dataset.tosparse()

    def _set_tested_classes(self):
        self.classes = self.dataset.classes


if __name__ == '__main__':
    CUnitTest.main()
