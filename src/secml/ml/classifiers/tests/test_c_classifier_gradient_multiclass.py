from secml.data.loader import CDLRandom
from secml.ml.classifiers import CClassifierSVM
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.utils import CUnitTest
from test_c_classifier_gradient import CClassifierGradientTestCases


class TestCClassifierGradientMulticlass(
    CClassifierGradientTestCases.TestCClassifierGradient):
    """Test of binary classifiers gradients."""

    def _dataset_creation(self):
        # generate synthetic data
        self.dataset = CDLRandom(n_features=3, n_redundant=0, n_informative=3,
                                 n_clusters_per_class=1, random_state=1).load()

        self.dataset_sparse = self.dataset.tosparse()

    def _clfs_creation(self):
        self.clfs = [CClassifierMulticlassOVA(CClassifierSVM)]
        self.clf_ids = ['OVA']

        for clf in self.clfs:  # Enabling debug output for each classifier
            clf.verbose = 2


if __name__ == '__main__':
    CUnitTest.main()
