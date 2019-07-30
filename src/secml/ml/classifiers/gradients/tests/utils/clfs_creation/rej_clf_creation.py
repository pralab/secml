from secml.ml.features.normalization import CNormalizerMinMax
from secml.ml.classifiers.reject import CClassifierRejectThreshold
import os
from six.moves import range
from secml.ml.classifiers import CClassifierSVM
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.array import CArray
from secml.utils import fm
from secml.adv.seceval import CSecEval
from secml.adv.attacks.evasion import CAttackEvasion
from secml.ml.kernel import CKernelRBF


def _generate_advx(dataset, clf_norej):
    adv_file = os.path.dirname(os.path.abspath(__file__))

    adv_file += "/" + 'adv_x.txt'

    if fm.file_exist(adv_file):
        adv_dts_X = CArray.load(adv_file)
    else:

        lb = dataset.X.min()
        ub = dataset.X.max()

        dmax_lst = [0.1, 0.2]
        discrete = False
        type_dist = 'l2'
        solver_type = 'gradient-bls'
        solver_params = {'eta': 0.1}

        params = {
            "classifier": clf_norej,
            "surrogate_classifier": clf_norej,
            "surrogate_data": dataset,
            "distance": type_dist,
            "dmax": 0,
            "lb": lb,
            "ub": ub,
            "discrete": discrete,
            "attack_classes": 'all',
            "y_target": None,
            "solver_type": solver_type,
            "solver_params": solver_params
        }

        evasion = CAttackEvasion(**params)
        sec_eval = CSecEval(attack=evasion,
                            param_name='dmax',
                            param_values=dmax_lst,
                            save_adv_ds=True)
        sec_eval.run_sec_eval(dataset)

        adv_dts_lst = sec_eval.sec_eval_data.adv_ds

        for adv_dts_idx in range(len(adv_dts_lst)):
            if adv_dts_idx > 0:
                if adv_dts_idx == 1:
                    adv_dts_X = adv_dts_lst[adv_dts_idx].X
                else:
                    adv_dts_X = adv_dts_X.append(
                        adv_dts_lst[adv_dts_idx].X)

            # save the computed adversarial examples
        adv_dts_X.save(adv_file)

    return adv_dts_X


def rej_clf_creation(clf_idx, normalizer=False, dataset=None):
    # Create the classifier with reject options needed to perform some tests

    if clf_idx == 'reject-threshold':
        kernel = CKernelRBF(gamma=1)
        clf = CClassifierMulticlassOVA(
            classifier=CClassifierSVM, class_weight='balanced',
            preprocess=None, kernel=kernel)
        clf.verbose = 0
        clf = CClassifierRejectThreshold(clf, 0.6)

    else:
        raise ValueError("classifier idx not managed!")

    if normalizer:
        normalizer = CNormalizerMinMax((-10, 10))
        clf.preprocess = normalizer

    return clf
