from secml.testing import CUnitTest

try:
    import cleverhans
except ImportError:
    CUnitTest.importskip("cleverhans")

from cleverhans.attacks import CarliniWagnerL2

from secml.array import CArray
from secml.data.loader import CDLRandomBlobs
from secml.figure import CFigure
from secml.ml.classifiers import CClassifierSVM
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.adv.attacks import CAttackEvasionCleverhans
from secml.ml.features import CNormalizerMinMax
from secml.ml.kernel import CKernelRBF


class TestEvasion2dDatasetCleverhans(CUnitTest):
    """Test our CleverHans evasion attack wrapper."""

    def _param_setter(self):
        # dataset
        self.n_features = 0
        self.n_samples = 100
        self.centers = [[0.1, 0.1], [0.5, 0], [0.8, 0.8]]
        self.cluster_std = 0.01

        self.y_target = 2

        self.nmz = CNormalizerMinMax()
        self.classifier = CClassifierMulticlassOVA(CClassifierSVM,
                                                   kernel=CKernelRBF(gamma=10),
                                                   C=0.1,
                                                   preprocess=self.nmz)

        self.lb = 0.0
        self.ub = 1.0

        self.name_file = 'CW_evasion2D_target_{}.png'.format(self.y_target)

    def setUp(self):
        # Set as true to save data required by `visualization_script`
        self.save_info_for_plot = False

        self.seed = 0

        self.logger.info("seed: {:}".format(str(self.seed)))

        self._param_setter()
        self.ds = CDLRandomBlobs(n_features=self.n_features,
                                 centers=self.centers,
                                 cluster_std=self.cluster_std,
                                 n_samples=self.n_samples,
                                 random_state=self.seed).load()

        self.logger.info("training classifier ...")
        self.classifier.fit(self.ds)
        self.logger.info("training classifier ... Done.")

        self._x0, self._y0 = CArray([0.2, 0.4]), CArray([0])

        attack_params = {
            'binary_search_steps': 1,
            'initial_const': 0.5,
            'confidence': 1,
            'abort_early': True,
            'clip_min': self.lb,
            'clip_max': self.ub,
            'max_iterations': 300,
            'learning_rate': 0.03,
        }
        self.attack = CAttackEvasionCleverhans(
            classifier=self.classifier,
            surrogate_classifier=self.classifier,
            surrogate_data=self.ds,
            y_target=self.y_target,
            n_classes=self.classifier.n_classes,
            n_feats=self.classifier.n_features,
            clvh_attack_class=CarliniWagnerL2,
            **attack_params)

    def test_plot(self):
        if self.save_info_for_plot:
            fig = CFigure()

            y_pred_CH, _, adv_ds_CH, _ = self.attack.run(self._x0, self._y0)
            fig.sp.plot_path(self.attack.x_seq)
            fig.sp.plot_fun(self.attack.objective_function, plot_levels=False,
                            multipoint=True, n_grid_points=50)
            fig.sp.plot_decision_regions(self.classifier,
                                         plot_background=False,
                                         n_grid_points=200)

            fig.title("y_target: {}".format(self.y_target))
            fig.savefig(self.name_file)


if __name__ == '__main__':
    CUnitTest.main()
