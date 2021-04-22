from secml.adv.attacks.evasion.tests import CAttackEvasionTestCases

from secml.adv.attacks.evasion import CAttackEvasionPGDLS
from secml.adv.seceval import CSecEval
from secml.array import CArray
from secml.figure import CFigure
from secml.ml.classifiers import CClassifierSVM


class TestCSecEval(CAttackEvasionTestCases):
    """Unittests for CSecEval (evasion attack)."""

    def setUp(self):

        self.clf = CClassifierSVM(C=1.0)

        self.n_tr = 40
        self.n_features = 10
        self.seed = 0

        self.logger.info(
            "Loading `random_blobs` with seed: {:}".format(self.seed))
        self.ds = self._load_blobs(
            self.n_features, 2, sparse=False, seed=self.seed)

        self.tr = self.ds[:self.n_tr, :]
        self.ts = self.ds[self.n_tr:, :]

        self.clf.fit(self.tr.X, self.tr.Y)

    def test_attack_pgd_ls(self):
        """Test SecEval using CAttackEvasionPGDLS."""
        params = {
            "classifier": self.clf,
            "double_init_ds": self.tr,
            "distance": 'l2',
            "lb": -2,
            "ub": 2,
            "y_target": None,
            "solver_params": {'eta': 0.1, 'eps': 1e-2}
        }
        attack = CAttackEvasionPGDLS(**params)
        attack.verbose = 1

        param_name = 'dmax'

        self._set_and_run(attack, param_name)

    def test_attack_pgd_ls_discrete(self):
        """Test SecEval using CAttackEvasionPGDLS on a problematic
        discrete case with L1 constraint.
        We alter the classifier so that many weights have the same value.
        The optimizer should be able to evade the classifier anyway,
        by changing one feature each iteration. Otherwise, by changing
        all the feature with the same value at once, the evasion will always
        fail because the L1 constraint will be violated.
        """
        self.ds = self._discretize_data(self.ds, eta=1)
        self.ds.X[self.ds.X > 1] = 1
        self.ds.X[self.ds.X < -1] = -1

        self.tr = self.ds[:self.n_tr, :]
        self.ts = self.ds[self.n_tr:, :]

        self.clf.fit(self.tr.X, self.tr.Y)

        # Set few features to the same max value
        w_new = self.clf.w.deepcopy()
        w_new[CArray.randint(
            self.clf.w.size, shape=5, random_state=0)] = self.clf.w.max()
        self.clf._w = w_new

        params = {
            "classifier": self.clf,
            "double_init": False,
            "distance": 'l1',
            "lb": -1,
            "ub": 1,
            "y_target": None,
            "solver_params": {'eta': 1, 'eps': 1e-2}
        }
        attack = CAttackEvasionPGDLS(**params)
        attack.verbose = 1

        param_name = 'dmax'

        self._set_and_run(attack, param_name, dmax_step=1)

    def test_attack_cleverhans(self):
        """Test SecEval using CAttackEvasionCleverhans+FastGradientMethod."""
        try:
            import cleverhans
        except ImportError as e:
            import unittest
            raise unittest.SkipTest(e)

        from cleverhans.attacks import FastGradientMethod
        from secml.adv.attacks import CAttackEvasionCleverhans
        params = {
            "classifier": self.clf,
            "surrogate_data": self.tr,
            "y_target": None,
            "clvh_attack_class": FastGradientMethod,
            'eps': 0.1,
            'clip_max': 2,
            'clip_min': -2,
            'ord': 2
        }
        attack = CAttackEvasionCleverhans(**params)

        param_name = 'attack_params.eps'

        self._set_and_run(attack, param_name)

    def _set_and_run(self, attack, param_name, dmax=2, dmax_step=0.5):
        """Create the SecEval and run it on test set."""
        param_values = CArray.arange(
            start=0, step=dmax_step,
            stop=dmax + dmax_step)

        sec_eval = CSecEval(
            attack=attack,
            param_name=param_name,
            param_values=param_values,
        )

        sec_eval.run_sec_eval(self.ts)

        self._plot_sec_eval(sec_eval)

        # At the end of the seceval we expect 0% accuracy
        self.assertFalse(
            CArray(sec_eval.sec_eval_data.Y_pred[-1] == self.ts.Y).any())

    @staticmethod
    def _plot_sec_eval(sec_eval):

        figure = CFigure(height=5, width=5)

        figure.sp.plot_sec_eval(sec_eval.sec_eval_data,
                                label='SVM', marker='o',
                                show_average=True, mean=True)

        figure.sp.title(sec_eval.attack.__class__.__name__)
        figure.subplots_adjust()
        figure.show()

    if __name__ == '__main__':
        CAttackEvasionTestCases.main()
