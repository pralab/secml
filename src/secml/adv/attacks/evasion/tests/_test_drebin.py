from __future__ import print_function

from secml.adv.attacks import CAttackEvasionBLS
from secml.adv.seceval import CSecEval

from secml.array import CArray
from secml.data.loader import CDataLoaderDrebinTDSC
from secml.ml.features.selection import CFeatSel
from secml.ml.classifiers import CClassifierSVM
from secml.utils import fm, pickle_utils

# ATTENTION!
# NB: THIS IS A DRAFT OF A TEST USING DREBIN (BOOLEAN FEATURES)
# FIXME: THIS SCRIPT IMPORTS DREBIN, WHICH IS AN INTERNAL-ONLY DATASET.
#  SHOULD BE REMOVED FROM MAIN BRANCH


if False or not fm.file_exist('ds.gz'):
    print("Creating the dataset")

    dl = CDataLoaderDrebinTDSC()
    dl.verbose = 1
    ds = dl.load(feats_info=False)

    tr = ds[:30000, :]
    ts = ds[30000:, :]

    selector = CFeatSel('standard')
    feats_idx = selector.selection(tr)[0]
    feats_idx = feats_idx[:5000]

    tr = tr[:, feats_idx]
    ts = ts[:, feats_idx]
    pickle_utils.save('ds.gz', (tr, ts))
else:
    tr, ts = pickle_utils.load('ds.gz', encoding='latin1')

mal_idx = ts.Y.find(ts.Y == 1)[:3]
adv_ds = ts[mal_idx, :]

if False or not fm.file_exist('clf.gz'):
    print("Training the classifier")

    clf = CClassifierSVM()
    clf.verbose = 1
    clf.fit(tr)
    pickle_utils.save('clf.gz', clf)
else:
    clf = pickle_utils.load('clf.gz', encoding='latin1')

solver_type = 'gradient-bls'
solver_params = {'eta': 1, 'eta_min': 1, 'eta_max': None, 'eps': 1e-4}
lb = 'x0'  # None
ub = 1     # None
distance = 'l1'
discrete = True
dmax = 5
dmax_step = 1

y_target = None


params = {
    "classifier": clf,
    "surrogate_classifier": clf,
    "surrogate_data": tr,
    "distance": distance,
    "lb": lb,
    "ub": ub,
    "discrete": discrete,
    "attack_classes": 'all',
    "y_target": y_target,
    "solver_type": solver_type,
    "solver_params": solver_params
}

evasion = CAttackEvasionBLS(**params)
evasion.verbose = 1

param_name = 'dmax'
param_values = CArray.arange(
    start=0, step=dmax_step, stop=dmax + dmax_step)

sec_eval = CSecEval(
    attack=evasion,
    param_name=param_name,
    param_values=param_values,
    save_adv_ds=True)


sec_eval.run_sec_eval(adv_ds)

print(sec_eval.sec_eval_data.adv_ds[0].X[0, :] != sec_eval.sec_eval_data.adv_ds[-1].X[0, :])