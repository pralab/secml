import os

from secml.utils import CUnitTest
from secml.ml.classifiers.reject.tests import CClassifierRejectTestCases

from secml.adv.attacks import CAttackEvasion
from secml.adv.defenses import CClassifierRejectDetector
from secml.adv.seceval import CSecEval
from secml.array import CArray
from secml.data.loader import CDLRandomBlobs
from secml.utils import fm


class TestCClassifierRejectDetector(
    CClassifierRejectTestCases.TestCClassifierReject):
    """Unit test for CClassifierRejectDetector"""

    def setUp(self):
        """Test for init and fit methods."""
        # generate synthetic data
        self.dataset = CDLRandomBlobs(n_features=2, n_samples=100,
                                      centers=((0, 0), (10, 10)),
                                      cluster_std=5.0, random_state=0).load()

        self.lb = self.dataset.X.min(axis=0)
        self.ub = self.dataset.X.max(axis=0)

        self.logger.info("Testing classifier creation ")

        from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
        from secml.ml.classifiers import CClassifierSVM
        self.clf_norej = CClassifierMulticlassOVA(CClassifierSVM)

        self.clf_norej.fit(self.dataset)

        det = CClassifierSVM(kernel='rbf')

        self._set_eva_params()

        self.adv_x = self._generate_advx()

        self.clf = CClassifierRejectDetector(
            self.clf_norej, det=det, adv_x=self.adv_x)
        self.clf.verbose = 2  # Enabling debug output for each classifier
        self.clf.fit(self.dataset)

    def _set_eva_params(self):

        self.dmax_lst = [5.5, 6.0]
        self.discrete = False
        self.type_dist = 'l2'
        self.solver_type = 'gradient-descent'
        self.solver_params = {'eta': 0.1}

    def _generate_advx(self):

        self.adv_file = os.path.dirname(os.path.abspath(__file__))

        self.adv_file += "/" + 'adv_x.txt'

        if fm.file_exist(self.adv_file):
            adv_dts_X = CArray.load(self.adv_file)
        else:
            params = {
                "classifier": self.clf_norej,
                "surrogate_classifier": self.clf_norej,
                "surrogate_data": self.dataset,
                "distance": self.type_dist,
                "dmax": 0,
                "lb": self.lb,
                "ub": self.ub,
                "discrete": self.discrete,
                "attack_classes": 'all',
                "y_target": None,
                "solver_type": self.solver_type,
                "solver_params": self.solver_params
            }

            self.evasion = CAttackEvasion(**params)
            self.sec_eval = CSecEval(attack=self.evasion, param_name='dmax',
                                     param_values=self.dmax_lst,
                                     save_adv_ds=True)
            self.sec_eval.run_sec_eval(self.dataset)

            adv_dts_lst = self.sec_eval.sec_eval_data.adv_ds

            for adv_dts_idx in xrange(len(adv_dts_lst)):
                if adv_dts_idx > 0:
                    if adv_dts_idx == 1:
                        adv_dts_X = adv_dts_lst[adv_dts_idx].X
                    else:
                        adv_dts_X = adv_dts_X.append(
                            adv_dts_lst[adv_dts_idx].X)

                # save the computed adversarial examples
            adv_dts_X.save(self.adv_file)

        return adv_dts_X


if __name__ == '__main__':
    CUnitTest.main()
