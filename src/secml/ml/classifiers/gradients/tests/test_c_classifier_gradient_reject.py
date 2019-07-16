from secml.testing import CUnitTest

from secml.ml.classifiers.gradients.tests.utils.clfs_creation.rej_clf_creation \
    import rej_clf_creation

from secml.data.loader import CDLRandom
from secml.ml.classifiers.gradients.tests.test_c_classifier_gradient import \
    CClassifierGradientTestCases


class TestCClassifierGradientReject(
    CClassifierGradientTestCases.TestCClassifierGradient):
    """Test of multiclass classifiers gradients."""

    @property
    def clf_list(self):
        return ['reject-threshold']  # , 'reject-detector']
        # fixme: read those classifier as soon as we will have fix their
        #  gradients
        # 'reject-detector'

    @property
    def clf_creation_function(self):
        return rej_clf_creation

    def _dataset_creation(self):
        # generate synthetic data
        self.dataset = CDLRandom(n_samples=100, n_classes=3, n_features=2,
                                 n_redundant=0, n_clusters_per_class=1,
                                 class_sep=1, random_state=0).load()

        # Add a new class modifying one of the existing clusters
        self.dataset.Y[(self.dataset.X[:, 0] > 0).logical_and(
            self.dataset.X[:, 1] > 1).ravel()] = self.dataset.num_classes

        self.lb = 0
        self.ub = 1

        self.dataset_sparse = self.dataset.tosparse()

    def _set_tested_classes(self):
        self.classes = self.dataset.classes.append(-1)


if __name__ == '__main__':
    CUnitTest.main()
