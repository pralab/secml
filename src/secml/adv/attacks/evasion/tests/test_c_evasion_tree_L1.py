from secml.testing import CUnitTest
from secml.adv.attacks.evasion.tests import CEvasionTestCases

from secml.ml.classifiers import CClassifierDecisionTree, CClassifierSVM
from secml.optim.constraints import CConstraintBox


class TestEvasionTreeL1(CEvasionTestCases.TestCEvasion):
    """Evasion with Tree classifier and L1 distance constraint."""

    def param_setter(self):

        self.type_dist = 'l1'

        self.dmax = 2  # On un-normalized data

        self.discrete = False
        self.eta = 1.0
        self.eta_min = None
        self.eta_max = None

        self.normalizer = None

        self.classifier = CClassifierDecisionTree()

        self.surrogate_classifier = CClassifierSVM(kernel='rbf')

        self.seed = 2333  # Random state generator for the dataset

        self.n_clusters = 2  # Number of dataset clusters
        self.n_features = 2  # Number of dataset features

        self.lb = -1.5
        self.ub = +1.5

        self.grid_limits = [(-2, 2), (-2, 2)]
        self.filename = 'test_c_evasion_tree_L1.pdf'

    def _plot_dataset_clf_and_box(self, evas, fig):
        """Plot dataset and box constraints"""
        # Overriding as the threshold level of decision tree is 0.5
        fig.switch_sptype(sp_type="ds")
        fig.sp.plot_ds(self.dataset)
        fig.switch_sptype(sp_type="function")
        fig.sp.plot_fobj(
            func=evas.classifier.decision_function,
            grid_limits=self.grid_limits, colorbar=False,
            levels=[0.5], y=1)
        # construct and plot box
        if self.lb == "x0":
            self.lb = self.x0
        if self.ub == "x0":
            self.ub = self.x0
        box = CConstraintBox(lb=self.lb, ub=self.ub)
        fig.sp.plot_fobj(func=box.constraint,
                         plot_background=False,
                         n_grid_points=20,
                         grid_limits=self.grid_limits,
                         levels=[0], colorbar=False)


if __name__ == '__main__':
    CUnitTest.main()
