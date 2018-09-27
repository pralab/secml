from secml.utils import CUnitTest
from test_c_evasion import CEvasionTestCases

from secml.classifiers import CClassifierDecisionTree
from secml.utils import fm


class TestEvasionTreeL1(CEvasionTestCases.TestCEvasion):
    """Evasion with Tree classifier and L1 distance constraint."""

    def param_setter(self):

        self.type_dist = 'l1'

        self.dmax = 6.4  # On un-normalized data

        self.discrete = False
        self.eta = 1.0
        self.eta_min = None
        self.eta_max = None

        self.normalizer = None

        self.classifier = CClassifierDecisionTree()

        self.seed = None  # Random state generator for the dataset

        self.n_clusters = 2  # Number of dataset clusters
        self.n_features = 2  # Number of dataset features

        self.lb = -5
        self.ub = +5

        self.grid_limits = [(-10, 10), (-10, 10)]
        self.name_file = fm.join(fm.abspath(__file__), 'L1_tree.pdf')


if __name__ == '__main__':
    CUnitTest.main()
