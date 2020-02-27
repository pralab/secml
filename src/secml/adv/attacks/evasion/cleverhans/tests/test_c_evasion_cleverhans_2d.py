from cleverhans.attacks import ElasticNetMethod, CarliniWagnerL2, \
    ProjectedGradientDescent, SPSA

from secml.testing import CUnitTest

try:
    import cleverhans
except ImportError:
    CUnitTest.importskip("cleverhans")

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

        self.y_target = None

        self.nmz = CNormalizerMinMax()
        self.classifier = CClassifierMulticlassOVA(
            CClassifierSVM, kernel=CKernelRBF(gamma=10),
            C=0.1, preprocess=self.nmz)

        self.lb = 0.0
        self.ub = 1.0

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

        self._x0 = CArray([0.6, 0.2])
        self._y0 = CArray(self.classifier.predict(self._x0))

    def test_SPSA(self):
        attack_params = {
            'eps': 0.5,
            'delta': 0.1,
            'clip_min': self.lb,
            'clip_max': self.ub,
            'nb_iter': 50,
            'learning_rate': 0.03,
        }
        self.attack = CAttackEvasionCleverhans(
            classifier=self.classifier,
            surrogate_classifier=self.classifier,
            surrogate_data=self.ds,
            y_target=self.y_target,
            clvh_attack_class=SPSA,
            **attack_params)
        self.y_pred_CH, _, self.adv_ds_CH, _ = self.attack.run(
            self._x0, self._y0)
        self._test_confidence()
        self._test_plot()

    def test_pgd(self):
        attack_params = {
            'eps': 0.5,
            'eps_iter': 0.1,
            'ord': 2,
            'rand_init': False,
            'nb_iter': 20
        }
        self.attack = CAttackEvasionCleverhans(
            classifier=self.classifier,
            surrogate_classifier=self.classifier,
            surrogate_data=self.ds,
            y_target=self.y_target,
            clvh_attack_class=ProjectedGradientDescent,
            **attack_params)
        self.y_pred_CH, _, self.adv_ds_CH, _ = self.attack.run(
            self._x0, self._y0)
        self._test_confidence()
        self._test_plot()

    def test_cw(self):
        attack_params = {
            'binary_search_steps': 4,
            'initial_const': 0.01,
            'confidence': 10,
            'abort_early': True,
            'clip_min': self.lb,
            'clip_max': self.ub,
            'max_iterations': 30,
            'learning_rate': 0.03,
        }
        self.attack = CAttackEvasionCleverhans(
            classifier=self.classifier,
            surrogate_classifier=self.classifier,
            surrogate_data=self.ds,
            y_target=self.y_target,
            clvh_attack_class=CarliniWagnerL2,
            **attack_params)
        self.y_pred_CH, _, self.adv_ds_CH, _ = self.attack.run(
            self._x0, self._y0)
        self._test_confidence()
        self._test_plot()
        self._test_stored_consts()

    def test_enmethod(self):
        attack_params = {
            'binary_search_steps': 3,
            'initial_const': 0.01,
            'confidence': 10,
            'abort_early': True,
            'clip_min': self.lb,
            'clip_max': self.ub,
            'max_iterations': 30,
            'learning_rate': 0.03,
        }
        self.attack = CAttackEvasionCleverhans(
            classifier=self.classifier,
            surrogate_classifier=self.classifier,
            surrogate_data=self.ds,
            y_target=self.y_target,
            decision_rule='END',
            clvh_attack_class=ElasticNetMethod,
            **attack_params)
        self.y_pred_CH, _, self.adv_ds_CH, _ = self.attack.run(
            self._x0, self._y0)
        self._test_confidence()
        self._test_plot()
        self._test_stored_consts()

    def _test_confidence(self):
        init_pred, init_score = self.classifier.predict(
            self._x0, return_decision_function=True)
        final_pred, final_score = self.classifier.predict(
            self.adv_ds_CH.X, return_decision_function=True)
        if self.y_target is not None:
            self.assertGreater(final_score[:, self.y_target].item(),
                               init_score[:, self.y_target].item())
        self.assertLess(final_score[self._y0].item(),
                        init_score[self._y0].item())

    def _test_plot(self):
        if self.save_info_for_plot:
            self.name_file = '{}_evasion2D_target_{}.png'.format(
                self.attack._clvrh_attack_class.__name__, self.y_target)
            fig = CFigure()

            fig.sp.plot_path(self.attack.x_seq)
            fig.sp.plot_fun(self.attack.objective_function, plot_levels=False,
                            multipoint=True, n_grid_points=50)
            fig.sp.plot_decision_regions(self.classifier,
                                         plot_background=False,
                                         n_grid_points=200)

            fig.title("ATTACK: {}, y_target: {}".format(
                self.attack._clvrh_attack_class.__name__, self.y_target))
            fig.savefig(self.name_file)
            fig.show()

    def _test_stored_consts(self):
        self.logger.info("Testing stored variables")
        self.assertTrue(len(self.attack.stored_vars.keys()) > 0)
        self.logger.info("Stored vars: {}".format(self.attack.stored_vars))


if __name__ == '__main__':
    CUnitTest.main()
