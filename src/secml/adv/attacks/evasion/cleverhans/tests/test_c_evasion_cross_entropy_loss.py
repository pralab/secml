from secml.testing import CUnitTest

try:
    import cleverhans
except ImportError:
    CUnitTest.importskip("cleverhans")

from cleverhans.attacks import ProjectedGradientDescent

from secml.array import CArray
from secml.data.loader import CDLRandomBlobs
from secml.figure import CFigure
from secml.ml.classifiers import CClassifierSVM
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.adv.attacks import CAttackEvasionCleverhans
from secml.ml.features import CNormalizerMinMax


class TestCAttackEvasionCrossEntropyLoss(CUnitTest):
    """Test our CleverHans cross-entropy-loss-based evasion attack wrapper."""

    def _param_setter(self):
        # dataset
        self.n_features = 0
        self.n_samples = 100
        self.centers = [[0.1, 0.1], [0.5, 0], [0.8, 0.8]]
        self.cluster_std = 0.01

        self.y_target = 2

        self.distance = 'l2'
        self.nmz = CNormalizerMinMax()
        self.classifier = CClassifierMulticlassOVA(CClassifierSVM,
                                                   # kernel=CKernelRBF(gamma=10),
                                                   # C=0.1,
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

        self._x0, self._y0 = CArray([0.8, 0.4]), CArray([1])

        attack_params = {
            'eps': 0.5,
            'eps_iter': 0.1,
            'ord': 2,
            'rand_init': False,
            'nb_iter': 2000
        }
        self.attack = CAttackEvasionCleverhans(
            classifier=self.classifier,
            surrogate_classifier=self.classifier,
            surrogate_data=self.ds,
            y_target=self.y_target,
            n_classes=self.classifier.n_classes,
            n_feats=self.classifier.n_features,
            clvh_attack_class=ProjectedGradientDescent,
            **attack_params)

    def test_attack(self):
        pred, scores = self.classifier.predict(self._x0, return_decision_function=True)
        y_adv_pred, y_adv_scores, adv_ds, f_obj = self.attack.run(self._x0, self._y0)
        self.assertLess(y_adv_scores[self._y0], scores[self._y0])

    def test_loss(self):
        loss = self.attack.objective_function(self._x0)
        self.logger.info("Loss: {}".format(loss))
        self.assertGreater(loss, 0)

    def test_plot(self):
        if self.save_info_for_plot:
            fig = CFigure()

            y_pred_CH, _, adv_ds_CH, _ = self.attack.run(self._x0, self._y0)
            fig.sp.plot_path(self.attack.x_seq)

            fig.sp.plot_fun(self.attack.objective_function, plot_levels=False,
                            multipoint=True, n_grid_points=200)
            fig.sp.plot_decision_regions(self.classifier,
                                         plot_background=False,
                                         n_grid_points=200)
            normalized_ds = self.ds.deepcopy()
            normalized_ds.X = self.nmz.transform(normalized_ds.X)
            fig.sp.plot_ds(self.ds)

            fig.title("y_target: {}".format(self.y_target))
            fig.savefig(self.name_file)


if __name__ == '__main__':
    CUnitTest.main()
