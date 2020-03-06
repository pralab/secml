from secml.adv.attacks.evasion.cleverhans.tests import \
    CAttackEvasionCleverhansTestCases

try:
    import cleverhans
except ImportError:
    CAttackEvasionCleverhansTestCases.importskip("cleverhans")

import tensorflow as tf

from cleverhans.attacks import ElasticNetMethod, CarliniWagnerL2, \
    ProjectedGradientDescent, SPSA

from secml.array import CArray
from secml.data.loader import CDLRandomBlobs
from secml.figure import CFigure
from secml.ml.classifiers import CClassifierSVM
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.adv.attacks import CAttackEvasionCleverhans
from secml.ml.features import CNormalizerMinMax
from secml.ml.kernels import CKernelRBF
from secml.utils import fm


class TestCAttackEvasionCleverhans(CAttackEvasionCleverhansTestCases):
    """Unittests for CAttackEvasionCleverhans."""
    make_figures = True  # Set as True to produce figures

    @classmethod
    def setUpClass(cls):

        CAttackEvasionCleverhansTestCases.setUpClass()

        cls.seed = 0

        cls.y_target = None

        cls.clf = CClassifierMulticlassOVA(
            CClassifierSVM, kernel=CKernelRBF(gamma=10),
            C=0.1, preprocess=CNormalizerMinMax())

        cls.ds = CDLRandomBlobs(n_features=0,
                                centers=[[0.1, 0.1], [0.5, 0], [0.8, 0.8]],
                                cluster_std=0.01,
                                n_samples=100,
                                random_state=cls.seed).load()

        cls.clf.fit(cls.ds)

        cls.x0 = CArray([0.6, 0.2])
        cls.y0 = CArray(cls.clf.predict(cls.x0))

    def test_SPSA(self):
        """Test of SPSA algorithm."""

        tf.set_random_seed(self.seed)

        attack_params = {
            'eps': 0.5,
            'delta': 0.1,
            'clip_min': 0.0,
            'clip_max': 1.0,
            'nb_iter': 50,
            'learning_rate': 0.03,
        }
        evas = CAttackEvasionCleverhans(
            classifier=self.clf,
            surrogate_classifier=self.clf,
            surrogate_data=self.ds,
            y_target=self.y_target,
            clvh_attack_class=SPSA,
            **attack_params)

        # FIXME: random seed not working for SPSA?
        self._run_evasion(evas, self.x0, self.y0, expected_x=None)

        self._test_confidence(
            self.x0, self.y0, evas.x_opt, self.clf, self.y_target)
        self._test_plot(evas)

    def test_PGD(self):
        """Test of ProjectedGradientDescent algorithm."""

        tf.set_random_seed(self.seed)

        attack_params = {
            'eps': 0.5,
            'eps_iter': 0.1,
            'ord': 2,
            'rand_init': False,
            'nb_iter': 20
        }
        evas = CAttackEvasionCleverhans(
            classifier=self.clf,
            surrogate_classifier=self.clf,
            surrogate_data=self.ds,
            y_target=self.y_target,
            clvh_attack_class=ProjectedGradientDescent,
            **attack_params)

        # Expected final optimal point
        expected_x = CArray([0.7643, 0.6722])

        self._run_evasion(evas, self.x0, self.y0, expected_x=expected_x)

        self._test_confidence(
            self.x0, self.y0, evas.x_opt, self.clf, self.y_target)
        self._test_plot(evas)

    def test_CWL2(self):
        """Test of CarliniWagnerL2 algorithm."""

        tf.set_random_seed(self.seed)

        attack_params = {
            'binary_search_steps': 4,
            'initial_const': 0.01,
            'confidence': 10,
            'abort_early': True,
            'clip_min': 0.0,
            'clip_max': 1.0,
            'max_iterations': 30,
            'learning_rate': 0.03,
        }
        evas = CAttackEvasionCleverhans(
            classifier=self.clf,
            surrogate_classifier=self.clf,
            surrogate_data=self.ds,
            y_target=self.y_target,
            clvh_attack_class=CarliniWagnerL2,
            **attack_params)

        # Expected final optimal point
        expected_x = CArray([0.8316, 0.5823])

        self._run_evasion(evas, self.x0, self.y0, expected_x=expected_x)

        self._test_confidence(
            self.x0, self.y0, evas.x_opt, self.clf, self.y_target)
        self._test_stored_consts(evas)
        self._test_plot(evas)

    def test_ENM(self):
        """Test of ElasticNetMethod algorithm."""

        tf.set_random_seed(self.seed)

        attack_params = {
            'binary_search_steps': 3,
            'initial_const': 0.01,
            'confidence': 10,
            'abort_early': True,
            'clip_min': 0.0,
            'clip_max': 1.0,
            'max_iterations': 30,
            'learning_rate': 0.03,
        }
        evas = CAttackEvasionCleverhans(
            classifier=self.clf,
            surrogate_classifier=self.clf,
            surrogate_data=self.ds,
            y_target=self.y_target,
            decision_rule='END',
            clvh_attack_class=ElasticNetMethod,
            **attack_params)

        # Expected final optimal point
        expected_x = CArray([0.7651, 0.7406])

        self._run_evasion(evas, self.x0, self.y0, expected_x=expected_x)

        self._test_confidence(
            self.x0, self.y0, evas.x_opt, self.clf, self.y_target)
        self._test_stored_consts(evas)
        self._test_plot(evas)

    def _test_stored_consts(self, evas):
        """Check if `stored_vars` is correctly populated.

        Parameters
        ----------
        evas : CAttackEvasionCleverhans

        """
        self.logger.info("Stored vars: {}".format(evas.stored_vars))
        self.assertTrue(len(evas.stored_vars.keys()) > 0)

    def _test_plot(self, evas):
        """Check if `stored_vars` is correctly populated.

        Parameters
        ----------
        evas : CAttackEvasionCleverhans

        """
        if self.make_figures is False:
            self.logger.debug("Skipping figures...")
            return

        fig = CFigure()

        fig.sp.plot_path(evas.x_seq)
        fig.sp.plot_fun(evas.objective_function,
                        plot_levels=False, multipoint=True,
                        n_grid_points=50)
        fig.sp.plot_decision_regions(self.clf,
                                     plot_background=False,
                                     n_grid_points=100)

        fig.title("ATTACK: {}, y_target: {}".format(
            evas._clvrh_attack_class.__name__, self.y_target))

        name_file = '{}_evasion2D_target_{}.pdf'.format(
            evas._clvrh_attack_class.__name__, self.y_target)
        fig.savefig(fm.join(self.images_folder, name_file), file_format='pdf')


if __name__ == '__main__':
    CAttackEvasionCleverhansTestCases.main()
