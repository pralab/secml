from secml.testing import CUnitTest

from secml.adv.seceval import CSecEval
from secml.adv.attacks.evasion import CAttackEvasionPGDLS
from secml.array import CArray
from secml.data.loader import CDLRandomBlobs
from secml.figure import CFigure
from secml.ml.classifiers import CClassifierSVM


class TestCSecEval(CUnitTest):
    """Unittests for CSecEval (evasion attack)."""

    def setUp(self):
        
        classifier = CClassifierSVM(
            kernel='linear', C=1.0, grad_sampling=1.0)

        # data parameters
        discrete = False

        lb = -2
        ub = +2

        n_tr = 20
        n_ts = 10
        n_features = 2
        
        n_reps = 1

        self.sec_eval = []
        self.attack_ds = []
        for rep_i in range(n_reps):

            self.logger.info(
                "Loading `random_blobs` with seed: {:}".format(rep_i))
            loader = CDLRandomBlobs(
                n_samples=n_tr + n_ts,
                n_features=n_features,
                centers=[(-0.5, -0.5), (+0.5, +0.5)],
                center_box=(-0.5, 0.5),
                cluster_std=0.5,
                random_state=rep_i * 100 + 10)
            ds = loader.load()

            tr = ds[:n_tr, :]
            ts = ds[n_tr:, :]
            
            classifier.fit(tr)
            
            self.attack_ds.append(ts)

            # only manipulate positive samples, targeting negative ones
            self.y_target = None
            attack_classes = CArray([1])
        
            params = {
                "classifier": classifier,
                "surrogate_classifier": classifier,
                "surrogate_data": tr,
                "distance": 'l1',
                "lb": lb,
                "ub": ub,
                "discrete": discrete,
                "y_target": self.y_target,
                "attack_classes": attack_classes,
                "solver_params": {'eta': 0.5, 'eps': 1e-2}
            }
            attack = CAttackEvasionPGDLS(**params)
            attack.verbose = 1
        
            # sec eval params
            param_name = 'dmax'
            dmax = 2
            dmax_step = 0.5
            param_values = CArray.arange(
                start=0, step=dmax_step,
                stop=dmax + dmax_step)

            # set sec eval object
            self.sec_eval.append(
                CSecEval(
                    attack=attack,
                    param_name=param_name,
                    param_values=param_values,
                    )
            )

    def _plot_sec_eval(self):
        # figure creation
        figure = CFigure(height=5, width=5)

        sec_eval_data = [
            sec_eval.sec_eval_data for sec_eval in self.sec_eval]
        # plot security evaluation
        figure.sp.plot_sec_eval(sec_eval_data, label='SVM', marker='o',
                                show_average=True, mean=True)

        figure.subplots_adjust()
        figure.show()

    def test_sec_eval(self):

        # evaluate classifier security
        for sec_eval_i, sec_eval in enumerate(self.sec_eval):
            sec_eval.run_sec_eval(self.attack_ds[sec_eval_i])

        self._plot_sec_eval()


if __name__ == '__main__':
    CUnitTest.main()
