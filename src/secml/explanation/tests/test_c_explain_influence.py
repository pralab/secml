from secml.explanation import CExplainerLocalInfluence
from secml.array import CArray
from secml.ml.classifiers import CClassifierLogistic, CClassifierRidge, CClassifierSVM
from secml.data.loader import CDataLoaderMNIST
from secml.data.splitter import CDataSplitterKFold


def create_mnist_dataset(digits=[4, 9], n_tr=50, n_val=1000, n_ts=1000,
                         seed=10):
    loader = CDataLoaderMNIST()

    tr = loader.load('training', digits=digits)

    ts = loader.load('testing', digits=digits, num_samples=n_ts)

    # start train and validation dataset split
    splitter = CDataSplitterKFold(num_folds=2, random_state=seed)
    splitter.compute_indices(tr)

    val_dts_idx = CArray.randsample(CArray.arange(0, tr.num_samples), n_val,
                                    random_state=seed)
    val = tr[val_dts_idx, :]

    tr_dts_idx = CArray.randsample(CArray.arange(0, tr.num_samples), n_tr,
                                   random_state=seed)
    tr = tr[tr_dts_idx, :]

    tr.X /= 255.0
    val.X /= 255.0
    ts.X /= 255.0

    return tr, val, ts

tr, val, ts = create_mnist_dataset()
#clf = CClassifierSVM()
#clf.store_dual_vars = True
#clf = CClassifierLogistic()
clf = CClassifierRidge()
clf.fit(tr)

clf_loss = clf.gradients._loss.class_type
explanation = CExplainerLocalInfluence(clf, tr, outer_loss_idx=clf_loss)

ts_sample_infl = explanation.explain(ts.X, ts.Y)

print ts_sample_infl.shape
