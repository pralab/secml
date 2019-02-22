from secml.utils import CUnitTest
from test_c_evasion import CEvasionTestCases

from secml.ml.classifiers import CClassifierSVM
from secml.ml.features.normalization import CNormalizerMinMax
from secml.utils import fm


class TestEvasionLinearL1(CEvasionTestCases.TestCEvasion):
    """Evasion with linear classifier and L1 distance constraint."""

    def param_setter(self):

        self.type_dist = 'l1'

        self.dmax = 5.0  # On un-normalized data

        self.discrete = True
        self.eta = 0.01
        self.eta_min = None
        self.eta_max = None

        self.normalizer = CNormalizerMinMax(feature_range=(-5, 5))
        self.classifier = CClassifierSVM(C=1.0, preprocess=self.normalizer)

        self.surrogate_classifier = self.classifier

        self.seed = 10  # Random state generator for the dataset

        self.sparse = True

        self.n_clusters = 2  # Number of dataset clusters
        self.n_features = 2  # Number of dataset features

        self.lb = -5.0
        self.ub = +5.0

        self.grid_limits = [(-5.5, 5.5), (-5.5, 5.5)]
        self.name_file = fm.join(fm.abspath(__file__), 'L1_linear.pdf')


if __name__ == '__main__':
    CUnitTest.main()
