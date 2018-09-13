import unittest
from test_c_evasion import TestCEvasionCases

from prlib.classifiers import CClassifierSVM
from prlib.features.normalization import CNormalizerMinMax
from prlib.kernel import CKernel


class TestEvasionNonLinearL2(TestCEvasionCases.TestCEvasion):
    """Evasion with nonlinear differentiable classifier
    and L2 distance constraint."""

    def param_setter(self):
        self.type_dist = 'l2'
        self.sparse = False

        self.dmax = 1.0

        self.discrete = False
        self.eta = 0.1
        self.eta_min = 0.1
        self.eta_max = None

        self.normalizer = CNormalizerMinMax(feature_range=(-1, 1))
        self.classifier = CClassifierSVM(
            kernel=CKernel.create('max', gamma=1.0), C=1.0,
            normalizer=self.normalizer)
        self.classifier.gamma = 0.01
        # self.classifier = CClassifierKDE(kernel='rbf', normalizer='minmax')

        self.seed = None  # Random state generator for the dataset
        # self.seed = 879858889
        # self.seed = 308757615

        self.n_clusters = 2  # Number of dataset clusters
        self.n_features = 2  # Number of dataset features

        self.lb = -2
        self.ub = 2

        self.grid_limits = [(-2.5, 2.5), (-2.5, 2.5)]
        self.name_file = 'L2_nonlinear.pdf'


if __name__ == '__main__':
    unittest.main()
