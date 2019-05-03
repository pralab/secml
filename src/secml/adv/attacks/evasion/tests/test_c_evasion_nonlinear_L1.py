from secml.testing import CUnitTest
from secml.adv.attacks.evasion.tests import CEvasionTestCases

from secml.ml.classifiers import CClassifierSVM
from secml.ml.features.normalization import CNormalizerMinMax


class TestEvasionNonLinearL1(CEvasionTestCases.TestCEvasion):
    """
    Evasion with nonlinear differentiable classifier
    and l1 distance constraint.
    """

    def param_setter(self):
        self.type_dist = 'l1'
        self.sparse = False  # sparse data support

        self.dmax = 1.0

        self.discrete = False
        self.eta = 0.1
        self.eta_min = 0.1
        self.eta_max = None

        self.normalizer = None

        normalizer = CNormalizerMinMax((-1, 1))
        self.classifier = CClassifierSVM(kernel='rbf', C=1,
                                         preprocess=normalizer)
        self.classifier.gamma = 2

        self.surrogate_classifier = self.classifier

        # self.seed = None  # Random state generator for the dataset
        self.seed = 87985889

        self.n_clusters = 2  # Number of dataset clusters
        self.n_features = 2  # Number of dataset features

        self.lb = -1.0
        self.ub = +1.0

        self.grid_limits = [(-1.5, 1.5), (-1.5, 1.5)]
        self.filename = 'test_c_evasion_nonlinear_L1.pdf'


if __name__ == '__main__':
    CUnitTest.main()
