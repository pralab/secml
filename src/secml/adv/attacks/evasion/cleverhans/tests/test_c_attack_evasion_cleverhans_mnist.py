from secml.adv.attacks.evasion.cleverhans.tests import \
    CAttackEvasionCleverhansTestCases

try:
    import cleverhans
except ImportError:
    CAttackEvasionCleverhansTestCases.importskip("cleverhans")

import tensorflow as tf

from cleverhans.attacks import FastGradientMethod, CarliniWagnerL2, \
    ElasticNetMethod, SPSA, LBFGS, \
    ProjectedGradientDescent, SaliencyMapMethod, \
    MomentumIterativeMethod, MadryEtAl, BasicIterativeMethod, DeepFool

from secml.array import CArray
from secml.data.loader import CDataLoaderMNIST
from secml.figure import CFigure
from secml.ml.classifiers import CClassifierSVM
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.utils import fm, pickle_utils
from secml.core.type_utils import is_float

from secml.adv.attacks.evasion import CAttackEvasionCleverhans


class TestCAttackEvasionCleverhansMNIST(CAttackEvasionCleverhansTestCases):
    """Unittests for CAttackEvasionCleverhans on MNIST dataset"""
    make_figures = False  # Set as True to produce figures

    @classmethod
    def setUpClass(cls):

        CAttackEvasionCleverhansTestCases.setUpClass()

        cls.seed = 0

        cls.tr, cls.val, cls.ts, cls.digits, \
            cls.img_w, cls.img_h = cls._load_mnist()

        cls.clf = CClassifierMulticlassOVA(CClassifierSVM)
        cls.clf.fit(cls.tr)
        
        cls.x0_img_class = 1
        cls.y_target = 2  # Target class for targeted tests

    @staticmethod
    def _load_mnist():
        """Load MNIST 4971 dataset."""
        digits = [4, 9, 7, 1]
        digits_str = "".join(['{:}-'.format(i) for i in digits[:-1]])
        digits_str += '{:}'.format(digits[-1])

        # FIXME: REMOVE THIS AFTER CDATALOADERS AUTOMATICALLY STORE DS
        tr_file = fm.join(
            fm.abspath(__file__), 'mnist_tr_{:}.gz'.format(digits_str))
        if not fm.file_exist(tr_file):
            loader = CDataLoaderMNIST()
            tr = loader.load('training', digits=digits)
            pickle_utils.save(tr_file, tr)
        else:
            tr = pickle_utils.load(tr_file, encoding='latin1')

        ts_file = fm.join(
            fm.abspath(__file__), 'mnist_ts_{:}.gz'.format(digits_str))
        if not fm.file_exist(ts_file):
            loader = CDataLoaderMNIST()
            ts = loader.load('testing', digits=digits)
            pickle_utils.save(ts_file, ts)
        else:
            ts = pickle_utils.load(ts_file, encoding='latin1')

        idx = CArray.arange(tr.num_samples)
        val_dts_idx = CArray.randsample(idx, 200, random_state=0)
        val_dts = tr[val_dts_idx, :]

        tr_dts_idx = CArray.randsample(idx, 200, random_state=0)
        tr = tr[tr_dts_idx, :]

        idx = CArray.arange(0, ts.num_samples)
        ts_dts_idx = CArray.randsample(idx, 200, random_state=0)
        ts = ts[ts_dts_idx, :]

        tr.X /= 255.0
        ts.X /= 255.0

        return tr, val_dts, ts, digits, tr.header.img_w, tr.header.img_h

    def _choose_x0_2c(self, x0_img_class):
        """Find a sample of that belong to the required class.
        
        Parameters
        ----------
        x0_img_class : int

        Returns
        -------
        x0 : CArray
        y0 : CArray
        
        """
        adv_img_idx = \
            CArray(self.ts.Y.find(self.ts.Y == x0_img_class))[0]

        x0 = self.ts.X[adv_img_idx, :]
        y0 = self.ts.Y[adv_img_idx]

        return x0, y0

    def test_DF(self):
        """Test of DeepFool algorithm."""
        attack = {'class': DeepFool,
                  'params': {'nb_candidate': 2,
                             'max_iter': 5,
                             'clip_min': 0.,
                             'clip_max': 1.0}}

        self._test_indiscriminate(attack)

    def test_FGM(self):
        """Test of FastGradientMethod algorithm."""
        attack = {'class': FastGradientMethod,
                  'params': {'eps': 0.3,
                             'clip_min': 0.,
                             'clip_max': 1.0}}

        self._test_targeted(attack)
        self._test_indiscriminate(attack)

    def test_ENM(self):
        """Test of ElasticNetMethod algorithm."""
        attack = {'class': ElasticNetMethod,
                  'params': {'max_iterations': 5,
                             'abort_early': True,
                             'learning_rate': 1e-3}}

        self._test_targeted(attack)
        self._test_indiscriminate(attack)

    def test_CWL2(self):
        """Test of CarliniWagnerL2 algorithm."""
        attack = {'class': CarliniWagnerL2,
                  'params': {'max_iterations': 5,
                             'learning_rate': 0.3,
                             'clip_min': 0.,
                             'clip_max': 1.0}}

        self._test_targeted(attack)
        self._test_indiscriminate(attack)

    def test_SPSA(self):
        """Test of SPSA algorithm."""
        attack = {'class': SPSA,
                  'params': {'eps': 0.5,
                             'nb_iter': 10,
                             'early_stop_loss_threshold': -1.,
                             'spsa_samples': 32,
                             'spsa_iters': 5,
                             'learning_rate': 0.03,
                             'clip_min': 0.,
                             'clip_max': 1., }}

        self._test_targeted(attack)
        # FIXME: random seed not working for SPSA?
        self._test_indiscriminate(attack, expected_y=None)

    def test_LBFGS(self):
        """Test of LBFGS algorithm."""
        attack = {'class': LBFGS,
                  'params': {'max_iterations': 5,
                             'clip_min': 0.,
                             'clip_max': 1., }}

        self._test_targeted(attack)

    def test_PGD(self):
        """Test of ProjectedGradientDescent algorithm."""
        attack = {'class': ProjectedGradientDescent,
                  'params': {'eps': 0.3,
                             'clip_min': 0.,
                             'clip_max': 1., }}

        self._test_targeted(attack)
        self._test_indiscriminate(attack)

    def test_SMM(self):
        """Test of SaliencyMapMethod algorithm."""
        attack = {'class': SaliencyMapMethod,
                  'params': {'clip_min': 0.,
                             'clip_max': 1., }}

        self._test_targeted(attack)

    def test_MIM(self):
        """Test of MomentumIterativeMethod algorithm."""
        attack = {'class': MomentumIterativeMethod,
                  'params': {'eps': 0.3,
                             'clip_min': 0.,
                             'clip_max': 1., }}

        self._test_targeted(attack)
        self._test_indiscriminate(attack)

    def test_Madry(self):
        """Test of MadryEtAl algorithm."""
        attack = {'class': MadryEtAl,
                  'params': {'eps': 0.3,
                             'clip_min': 0.,
                             'clip_max': 1., }}

        self._test_targeted(attack)
        self._test_indiscriminate(attack)

    def test_BIM(self):
        """Test of BasicIterativeMethod algorithm."""
        attack = {'class': BasicIterativeMethod,
                  'params': {'eps': 0.3,
                             'clip_min': 0.,
                             'clip_max': 1., }}

        self._test_targeted(attack)
        self._test_indiscriminate(attack)

    def _test_targeted(self, attack):
        """Perform a targeted attack.

        Parameters
        ----------
        attack : dict
            Dictionary with attack definition. Keys are "class",
            cleverhans attack class, and "params", dictionary
            with cleverhans attack parameters.

        """
        self._run(attack, y_target=self.y_target, expected_y=self.y_target)

    def _test_indiscriminate(self, attack, expected_y=0):
        """Perform an indiscriminate attack.

        Parameters
        ----------
        attack : dict
            Dictionary with attack definition. Keys are "class",
            cleverhans attack class, and "params", dictionary
            with cleverhans attack parameters.
        expected_y : int or CArray or None, optional
            Label of the expected final optimal point.
            Default 0 for the configuration defined in `setUpClass`.

        """
        self._run(attack, y_target=None, expected_y=expected_y)

    def _run(self, attack, y_target=None, expected_y=None):
        """Run evasion using input attack.

        Parameters
        ----------
        attack : dict
            Dictionary with attack definition. Keys are "class",
            cleverhans attack class, and "params", dictionary
            with cleverhans attack parameters.
        y_target : int or None
            Attack target class.
        expected_y : int or CArray or None, optional
            Label of the expected final optimal point.

        """
        attack_idx = attack['class'].__name__
        self.logger.info("Running algorithm: {:} ".format(attack_idx))

        tf.set_random_seed(self.seed)

        evas = CAttackEvasionCleverhans(
            classifier=self.clf,
            surrogate_classifier=self.clf,
            surrogate_data=self.val,
            y_target=y_target,
            clvh_attack_class=attack['class'],
            **attack['params']
        )

        evas.verbose = 2

        x0, y0 = self._choose_x0_2c(self.x0_img_class)
        
        with self.logger.timer():
            y_pred, scores, adv_ds, f_obj = evas.run(x0, y0)

        self.logger.info("Starting score: " + str(
            evas.classifier.decision_function(x0, y=1).item()))

        self.logger.info("Final score: " + str(evas.f_opt))
        self.logger.info("x*:\n" + str(evas.x_opt))
        self.logger.info("Point sequence:\n" + str(evas.x_seq))
        self.logger.info("Score sequence:\n" + str(evas.f_seq))
        self.logger.info("Fun Eval: " + str(evas.f_eval))
        self.logger.info("Grad Eval: " + str(evas.grad_eval))

        # Checking output
        self.assertEqual(1, y_pred.size)
        self.assertEqual(1, scores.shape[0])
        self.assertEqual(1, adv_ds.num_samples)
        self.assertEqual(adv_ds.issparse, x0.issparse)
        self.assertTrue(is_float(f_obj))

        if expected_y is not None:
            self.assert_array_almost_equal(y_pred.item(), expected_y)

        self._test_confidence(x0, y0, evas.x_opt, self.clf, y_target)
        self._show_adv(x0, y0, evas.x_opt, y_pred, attack_idx, y_target)

    def _show_adv(self, x0, y0, x_opt, y_pred, attack_idx, y_target):
        """Show the original and the modified sample.

        Parameters
        ----------
        x0 : CArray
            Initial attack point.
        y0 : CArray
            Label of the initial attack point.
        x_opt : CArray
            Final optimal point.
        y_pred : CArray
            Predicted label of the final optimal point.
        attack_idx : str
            Identifier of the attack algorithm.
        y_target : int or None
            Attack target class.

        """
        if self.make_figures is False:
            self.logger.debug("Skipping figures...")
            return

        added_noise = abs(x_opt - x0)  # absolute value of noise image

        fig = CFigure(height=5.0, width=15.0)

        fig.subplot(1, 3, 1)
        fig.sp.title(self.digits[y0.item()])
        fig.sp.imshow(x0.reshape((self.img_h, self.img_w)), cmap='gray')

        fig.subplot(1, 3, 2)
        fig.sp.imshow(
            added_noise.reshape((self.img_h, self.img_w)), cmap='gray')

        fig.subplot(1, 3, 3)
        fig.sp.title(self.digits[y_pred.item()])
        fig.sp.imshow(x_opt.reshape((self.img_h, self.img_w)), cmap='gray')

        name_file = "{:}_MNIST_target-{:}.pdf".format(attack_idx, y_target)
        fig.savefig(fm.join(self.images_folder, name_file), file_format='pdf')


if __name__ == '__main__':
    CAttackEvasionCleverhansTestCases.main()
