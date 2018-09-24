import unittest
from test_c_evasion import CEvasionTestCases

from secml.classifiers import CClassifierSVM
from secml.features.normalization import CNormalizerMinMax


class TestEvasionLinearL2(CEvasionTestCases.TestCEvasion):
    """Evasion with linear classifier and L2 distance constraint."""

    def param_setter(self):

        self.type_dist = 'l2'
        self.sparse = True

        self.dmax = 1.0  # On un-normalized data

        self.discrete = False
        self.eta = 0.5
        self.eta_min = None
        self.eta_max = None

        self.normalizer = CNormalizerMinMax(feature_range=(-1, 1))
        self.classifier = CClassifierSVM(C=0.1, normalizer=self.normalizer)

        self.seed = 48574308  # Random state generator for the dataset

        self.n_clusters = 2  # Number of dataset clusters
        self.n_features = 2  # Number of dataset features

        self.lb = -1.5
        self.ub = +1.5

        self.grid_limits = [(-1.5, 1.5), (-1.5, 1.5)]
        self.name_file = 'L2_linear.pdf'


if __name__ == '__main__':
    unittest.main()
