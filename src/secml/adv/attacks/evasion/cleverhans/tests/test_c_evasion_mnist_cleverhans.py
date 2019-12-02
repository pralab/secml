from secml.testing import CUnitTest

try:
    import cleverhans
except ImportError:
    CUnitTest.importskip("cleverhans")

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

from secml.adv.attacks.evasion import CAttackEvasionCleverhans


class TestEvasionMNISTCleverhansAttack(CUnitTest):
    """Test our CleverHans evasion attack wrapper."""

    def _param_setter(self):

        self.x0_img_class = 1
        self.y_target = None

        self.sparse = False
        self.distance = 'l2'
        self.dmax = 2

        self.eta = 1.0 / 255.0
        self.eta_min = 0.1
        self.eta_max = None

        self.classifier = CClassifierMulticlassOVA(CClassifierSVM)

        self.lb = 0.0
        self.ub = 1.0

        self.name_file = 'MNIST_evasion.pdf'

    def setUp(self):

        # Set as true to save data required by `visualization_script`
        self.save_info_for_plot = False

        self.seed = 0

        self.logger.info("seed: {:}".format(str(self.seed)))

        self._param_setter()
        self.tr, self.val, self.ts, self.digits, \
            self.img_w, self.img_h = self._load_mnist()

        # normalize in [lb,ub]
        self.tr.X *= (self.ub - self.lb)
        if self.lb != 0:
            self.tr.X += self.lb
        self.ts.X *= (self.ub - self.lb)
        if self.lb != 0:
            self.ts.X += self.lb

        if self.sparse is True:
            self.tr = self.tr.tosparse()
            self.ts = self.ts.tosparse()

        self.logger.info("Training classifier ...")
        self.classifier.fit(self.tr)
        self.logger.info("Training classifier ... Done.")
        self._chose_x0()

        # dictionary that contain the parameters of the cleverhans attack
        self.clvh_attacks = [

            {'class': DeepFool, 'params': {'nb_candidate': 2,
                                           'max_iter': 5,
                                           'clip_min': 0.,
                                           'clip_max': 1.0}},

            {'class': FastGradientMethod, 'params': {'eps': 0.3,
                                                     'clip_min': 0.,
                                                     'clip_max': 1.0}},

            {'class': ElasticNetMethod, 'params': {'max_iterations': 5,
                                                   'abort_early': True,
                                                   'learning_rate': 1e-3}},

            {'class': CarliniWagnerL2, 'params': {'max_iterations': 5,
                                                  'learning_rate': 0.3,
                                                  'clip_min': 0.,
                                                  'clip_max': 1.0}},

            {'class': SPSA, 'params': {'eps': 0.3,
                                       'nb_iter': 5,
                                       'early_stop_loss_threshold': -1.,
                                       'spsa_samples': 32,
                                       'spsa_iters': 5, 'is_debug': False,
                                       'clip_min': 0.,
                                       'clip_max': 1., }},

            {'class': LBFGS, 'params': {'max_iterations': 5,
                                        'clip_min': 0.,
                                        'clip_max': 1., }},

            {'class': ProjectedGradientDescent, 'params': {'eps': 0.3,
                                                           'clip_min': 0.,
                                                           'clip_max': 1., }},

            {'class': SaliencyMapMethod, 'params': {'clip_min': 0.,
                                                    'clip_max': 1., }},

            {'class': MomentumIterativeMethod, 'params': {'eps': 0.3,
                                                          'clip_min': 0.,
                                                          'clip_max': 1., }},

            {'class': MadryEtAl, 'params': {'eps': 0.3,
                                            'clip_min': 0.,
                                            'clip_max': 1., }},

            {'class': BasicIterativeMethod, 'params': {'eps': 0.3,
                                                       'clip_min': 0.,
                                                       'clip_max': 1., }},
        ]

    @staticmethod
    def _load_mnist():

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

    def _chose_x0(self):
        """Find a sample of that belong to the required class."""
        adv_img_idx = \
            CArray(self.ts.Y.find(self.ts.Y == self.x0_img_class))[0]

        self._x0 = self.ts.X[adv_img_idx, :]
        self._y0 = self.ts.Y[adv_img_idx]

    def _show_adv(self, x0, y0, xopt, y_pred):
        """Show the original and the modified sample.

        Parameters
        ----------
        x0 : original image
        xopt : modified sample

        """
        added_noise = abs(xopt - x0)  # absolute value of noise image

        if self.distance == 'l1':
            self.logger.info("Norm of input perturbation (l1): {:}".format(
                added_noise.ravel().norm(ord=1)))
        else:
            self.logger.info("Norm of input perturbation (l2): ".format(
                added_noise.ravel().norm()))

        fig = CFigure(height=5.0, width=15.0)
        fig.subplot(1, 3, 1)
        fig.sp.title(self.digits[y0])
        fig.sp.imshow(x0.reshape((self.img_h, self.img_w)), cmap='gray')
        fig.subplot(1, 3, 2)
        fig.sp.imshow(
            added_noise.reshape((self.img_h, self.img_w)), cmap='gray')
        fig.subplot(1, 3, 3)
        fig.sp.title(self.digits[y_pred[0]])
        fig.sp.imshow(xopt.reshape((self.img_h, self.img_w)), cmap='gray')
        fig.savefig(fm.join(
            fm.abspath(__file__), self.name_file), file_format='pdf')
        fig.show()

    def test_targeted_attack(self):
        """
        This test performs a targeted attack with all the attack classes
        specified in the dictionary clvh_attacks and check if the attack
        goes as we expected (namely if it really increase the score given by
        the classifier to the taraget class)
        """
        self.logger.info("Test targeted attacks")

        ids_untestable_clf = ['DeepFool']
        self._test_attack(y_target_idx=2, attack_type='targeted',
                          ids_untestable_clf=ids_untestable_clf)

    def test_indiscriminate_attack(self):
        """
        This test performs an indiscriminate attack with all the attack classes
        specified in the dictionary clvh_attacks and check if the attack
        goes as we expected (namely if it really decrease the score of the
        true class)
        """
        self.logger.info("Test indiscriminate attacks")

        ids_untestable_clf = ['LBFGS', 'SaliencyMapMethod']
        self._test_attack(y_target_idx=None, attack_type='indiscriminate',
                          ids_untestable_clf=ids_untestable_clf)

    def _test_attack(self, y_target_idx, attack_type, ids_untestable_clf):
        """
        Test if the attacks work checking if they are able to decrease the
        score given by the classifier to the true sample class.

        if the parameter "generate_plot" is true this function save also the
        images of the genearted attacks

        Parameters
        -------
        y_target_idx : int or None
            index of the target class in the list that contains the class
            labels

        """
        for atk_idx in range(len(self.clvh_attacks)):

            attack_idx = self.clvh_attacks[atk_idx]['class'].__name__

            if attack_idx not in ids_untestable_clf:

                self.logger.info("Run the {:} attack ".format(attack_idx))

                self._evasion_obj = CAttackEvasionCleverhans(
                    classifier=self.classifier,
                    surrogate_classifier=self.classifier,
                    n_feats=self.tr.num_features,
                    n_classes=self.tr.num_classes,
                    surrogate_data=self.val,
                    y_target=y_target_idx,
                    clvh_attack_class=self.clvh_attacks[atk_idx]['class'],
                    **self.clvh_attacks[atk_idx]['params']
                )

                self._evasion_obj.verbose = 2

                y_pred, scores, adv_ds = self._evasion_obj.run(
                    self._x0, self._y0)[:3]
                adv_x = adv_ds.X

                self.logger.info("num grad eval {:}".format(
                self._evasion_obj.grad_eval))
                self.logger.info("num f eval {:}".format(
                self._evasion_obj.f_eval))

                if self.save_info_for_plot:
                    info_dir = fm.join(fm.abspath(__file__), 'attacks_data')
                    if not fm.folder_exist(info_dir):
                        fm.make_folder(info_dir)

                    self._x0.save(
                        fm.join(info_dir, '{:}_x0'.format(attack_idx)),
                        overwrite=True)
                    adv_x.save(
                        fm.join(info_dir, '{:}_adv_x'.format(attack_idx)),
                        overwrite=True)
                    CArray(self._y0).save(
                        fm.join(info_dir, '{:}_y0'.format(attack_idx)),
                        overwrite=True)
                    CArray(y_pred).save(
                        fm.join(info_dir, '{:}_ypred'.format(attack_idx)),
                        overwrite=True)

                y0 = self._y0.item()

                # check if the attack works as expected
                if self.y_target:

                    s_ytarget_x0 = self.classifier.decision_function(
                        self._x0, self.y_target)
                    s_ytarget_xopt = self.classifier.decision_function(
                        adv_x, self.y_target)

                    self.logger.info(
                        "Discriminant function w.r.t the "
                        "target class first: {:} "
                        "and after evasion: {:}".format(
                            s_ytarget_x0,
                            s_ytarget_xopt))

                    self.assertLess(s_ytarget_x0, s_ytarget_xopt,
                                    "{:} attack {:} "
                                    "failed!".format(
                                        attack_type, attack_idx))

                else:  # indiscriminate attack

                    s_ytrue_x0 = self.classifier.decision_function(
                        self._x0, y0)
                    s_ytrue_xopt = self.classifier.decision_function(
                        adv_x, y0)

                    self.logger.info("Discriminant function w.r.t the "
                                     "true class first: {:} and after "
                                     "evasion: {:}".format(s_ytrue_x0,
                                                           s_ytrue_xopt))

                    self.assertGreater(
                        s_ytrue_x0, s_ytrue_xopt,
                        "{:} attack {:} failed!".format(
                            attack_type, attack_idx, ))


if __name__ == '__main__':
    CUnitTest.main()
