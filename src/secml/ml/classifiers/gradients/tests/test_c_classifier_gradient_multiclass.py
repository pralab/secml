from secml.testing import CUnitTest

from secml.data.loader import CDLRandom
from secml.ml.classifiers.gradients.tests import \
    CClassifierGradientTestCases, multiclass_clf_creation


class TestCClassifierGradientMulticlass(
    CClassifierGradientTestCases.TestCClassifierGradient):
    """Test of multiclass classifiers gradients."""

    @property
    def clf_list(self):
        return ['OVA']

    @property
    def clf_creation_function(self):
        return multiclass_clf_creation

    def _dataset_creation(self):
        # generate synthetic data
        self.dataset = CDLRandom(n_features=3, n_redundant=0, n_informative=3,
                                 n_clusters_per_class=1, random_state=1).load()

        self.dataset_sparse = self.dataset.tosparse()

    def _set_tested_classes(self):
        self.classes = self.dataset.classes


if __name__ == '__main__':
    CUnitTest.main()
