import unittest
from test_c_evasion import CEvasionTestCases

from secml.classifiers import CClassifierSVM, CClassifierDecisionTree
from secml.features.normalization import CNormalizerMinMax
from secml.kernel import CKernelRBF, CKernelLaplacian


class TestEvasionNonLinearL1(CEvasionTestCases.TestCEvasion):
    """
    Evasion with nonlinear differentiable classifier
    and l1 distance constraint.
    """

    def param_setter(self):
        self.type_dist = 'l1'
        self.sparse = True  # sparse data support

        self.dmax = 1.0

        self.discrete = False
        self.eta = 0.1
        self.eta_min = 0.3
        self.eta_max = None

        self.normalizer = None

        normalizer = CNormalizerMinMax((-1, 1))
        self.classifier = CClassifierSVM(kernel='rbf', C=1,
                                         normalizer=normalizer)
        self.classifier.gamma = 2

        # self.seed = None  # Random state generator for the dataset
        self.seed = 87985889

        self.n_clusters = 2  # Number of dataset clusters
        self.n_features = 2  # Number of dataset features

        self.lb = -1.0
        self.ub = +1.0

        self.grid_limits = [(-1.5, 1.5), (-1.5, 1.5)]
        self.name_file = 'L1_nonlinear.pdf'


if __name__ == '__main__':
    unittest.main()
