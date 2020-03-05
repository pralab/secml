from secml.testing import CUnitTest
from secml.ml.classifiers.gradients.tests.utils.clfs_creation import \
    binary_clf_creation

from secml.ml.classifiers.gradients.tests.test_c_classifier_gradient import \
    CClassifierGradientTestCases
from secml.data.loader import CDLRandom


class TestCClassifierGradientBinary(
    CClassifierGradientTestCases.TestCClassifierGradient):
    """Compare the analytical and the numerical gradient of binary
    classifiers."""

    @property
    def clf_list(self):
        return [
            'ridge', 'logistic', 'lin-svm', 'rbf-svm', 'sgd-lin', 'sgd-rbf']

    @property
    def clf_creation_function(self):

        # TODO: remove this filter when `kernel` parameter is removed from SGD Classifier
        self.logger.filterwarnings("ignore", message="`kernel` parameter.*",
                                   category=DeprecationWarning)

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
