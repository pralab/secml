from secml.testing import CUnitTest
from secml.adv.attacks.evasion.tests import CEvasionTestCases

from secml.ml.classifiers import CClassifierSVM
from secml.ml.features.normalization import CNormalizerMinMax


class TestEvasionLinearL2(CEvasionTestCases.TestCEvasion):
    """Evasion with linear classifier and L2 distance constraint."""

    def param_setter(self):
        self.type_dist = 'l2'
        self.sparse = True

        self.dmax = 1.05  # On un-normalized data

        self.discrete = False
        self.eta = 0.5
        self.eta_min = None
        self.eta_max = None

        self.normalizer = CNormalizerMinMax(feature_range=(-1, 1))
        self.classifier = CClassifierSVM(C=0.1, preprocess=self.normalizer)

        self.surrogate_classifier = self.classifier

        self.seed = 48574308  # Random state generator for the dataset

        self.n_clusters = 2  # Number of dataset clusters
        self.n_features = 2  # Number of dataset features

        self.lb = -0.67
        self.ub = +0.67

        self.grid_limits = [(-1.5, 1.5), (-1.5, 1.5)]
        self.filename = 'test_c_evasion_linear_L2.pdf'


if __name__ == '__main__':
    CUnitTest.main()
