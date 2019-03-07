import os

from secml.utils import CUnitTest
from secml.ml.classifiers.reject.tests import CClassifierRejectTestCases

from secml.adv.attacks import CAttackEvasion
from secml.adv.defenses import CClassifierRejectDetector
from secml.adv.seceval import CSecEval
from secml.array import CArray
from secml.data import CDataset
from secml.data.loader import CDLRandomBlobs
from secml.ml.kernel import CKernelRBF
from secml.ml.features import CPreProcess
from secml.utils import fm


class TestCClassifierRejectDetector(
    CClassifierRejectTestCases.TestCClassifierReject):
    """Unit test for CClassifierRejectDetector"""

    def setUp(self):
        """Test for init and fit methods."""
        # generate synthetic data
        self.dataset = CDLRandomBlobs(n_features=2, n_samples=50,
                                      centers=((-1, -1), (1, 1)),
                                      cluster_std=0.75, random_state=0).load()

        self.lb = self.dataset.X.min(axis=0)
        self.ub = self.dataset.X.max(axis=0)

        self.logger.info("Testing classifier creation ")

        from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
        from secml.ml.classifiers import CClassifierSVM
        self.clf_norej = CClassifierMulticlassOVA(CClassifierSVM)

        self.clf_norej.fit(self.dataset)

        det = CClassifierSVM(kernel=CKernelRBF(gamma=10))
        self._set_eva_params()

        self.adv_x = self._generate_advx()

        self.clf = CClassifierRejectDetector(
            self.clf_norej, det=det, adv_x=self.adv_x)
        self.clf.fit(self.dataset)

    def _set_eva_params(self):

        self.dmax_lst = [1, 1.5]
        self.discrete = False
        self.type_dist = 'l2'
        self.solver_type = 'descent-direction'
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
            self.evasion.verbose = 1
            self.sec_eval = CSecEval(attack=self.evasion, param_name='dmax',
                                     param_values=self.dmax_lst,
                                     save_adv_ds=True)
            self.sec_eval.verbose = 1
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

    def test_gradient(self):
        """Unittest for gradient_f_x method."""
        # Training the classifier
        clf = self.clf.fit(self.dataset)

        idx_test = 7
        x = self.dataset.X[idx_test, :]

        self.logger.info("Test pattern {:}:\n{:}".format(idx_test, x))

        self._test_gradient_numerical(
            clf, x, extra_classes=[-1], th=0.1, epsilon=0.01)

    def _create_preprocess_test(self, ds, clf, pre_id_list, kwargs_list):
        """Fit 2 clf, one with internal preprocessor chain
        and another using pre-transformed data."""
        pre1 = CPreProcess.create_chain(pre_id_list, kwargs_list)
        data_pre = pre1.fit_transform(ds.X)

        pre2 = CPreProcess.create_chain(pre_id_list, kwargs_list)
        clf_pre = clf.deepcopy()
        clf_pre.preprocess = pre2

        # We should preprocess adv_x too
        clf.adv_x = pre1.transform(clf.adv_x)

        clf_pre.fit(ds)
        clf.fit(CDataset(data_pre, ds.Y))

        return pre1, data_pre, clf_pre, clf

    def test_preprocess(self):
        """Test classifier with preprocessors inside."""
        # All linear transformations with gradient implemented
        self._test_preprocess(self.dataset, self.clf,
                              ['min-max', 'mean-std'],
                              [{'feature_range': (-1, 1)}, {}])
        self._test_preprocess_grad(self.dataset, self.clf,
                                   ['min-max', 'mean-std'],
                                   [{'feature_range': (-1, 1)}, {}],
                                   extra_classes=[-1],
                                   check_numerical=False)

        # Mixed linear/nonlinear transformations without gradient
        self._test_preprocess(
            self.dataset, self.clf, ['pca', 'unit-norm'], [{}, {}])


if __name__ == '__main__':
    CUnitTest.main()
