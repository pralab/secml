from secml.testing import CUnitTest
from secml.adv.attacks.evasion.tests import CEvasionTestCases

from secml.ml.classifiers import CClassifierSVM
from secml.ml.features.normalization import CNormalizerMinMax


class TestEvasionNonLinearL2(CEvasionTestCases.TestCEvasion):
    """Evasion with nonlinear differentiable classifier
    and L2 distance constraint."""

    def param_setter(self):
        self.type_dist = 'l2'
        self.sparse = False

        self.dmax = 1.25

        self.discrete = False
        self.eta = 0.01
        self.eta_min = 0.01
        self.eta_max = None

        self.normalizer = CNormalizerMinMax(feature_range=(-1, 1))
        self.classifier = CClassifierSVM(kernel='rbf', C=1,
                                         preprocess=self.normalizer)

        self.surrogate_classifier = self.classifier

        self.seed = 534513  # Random state generator for the dataset

        self.n_clusters = 2  # Number of dataset clusters
        self.n_features = 2  # Number of dataset features

        self.lb = -0.65
        self.ub = 1

        self.grid_limits = [(-2.5, 2.5), (-2.5, 2.5)]
        self.filename = 'test_c_evasion_nonlinear_L2.pdf'


if __name__ == '__main__':
    CUnitTest.main()
