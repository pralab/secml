import os
from six.moves import range

from secml.utils import CUnitTest
from secml.adv.attacks.evasion.tests.test_evasion_reject import \
    CEvasionRejectTestCases

from secml.adv.defenses import CClassifierRejectDetector
from secml.ml.classifiers import CClassifierSVM
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.array import CArray
from secml.utils import fm
from secml.adv.seceval import CSecEval
from secml.adv.attacks.evasion import CAttackEvasion
from secml.ml.kernel import CKernelRBF


class TestEvasionRejectDetector(CEvasionRejectTestCases.TestCEvasionReject):

    def _classifier_creation(self):
        # self.kernel = None
        self.kernel = CKernelRBF(gamma=1)

        self.clf_norej = CClassifierMulticlassOVA(
            classifier=CClassifierSVM, class_weight='balanced',
            preprocess=None, kernel=self.kernel)
        self.clf_norej.verbose = 0

        self.clf_norej.fit(self.ds)

        self._set_eva_params()
        adv_x = self._generate_advx()

        det = CClassifierSVM(kernel='rbf')
        self.multiclass = CClassifierRejectDetector(
            self.clf_norej, det=det, adv_x=adv_x)

    def _set_eva_params(self):

        self.dmax_lst = [0.1, 0.2]
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
                "surrogate_data": self.ds,
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
            self.sec_eval.run_sec_eval(self.ds)

            adv_dts_lst = self.sec_eval.sec_eval_data.adv_ds

            for adv_dts_idx in range(len(adv_dts_lst)):
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
