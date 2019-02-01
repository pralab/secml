from secml.array import CArray
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

    tr1 = tr[splitter.tr_idx[0], :]
    tr2 = tr[splitter.ts_idx[0], :]

    val_dts_idx = CArray.randsample(CArray.arange(0, tr1.num_samples), n_val,
                                    random_state=seed)
    val = tr1[val_dts_idx, :]

    tr_dts_idx = CArray.randsample(CArray.arange(0, tr1.num_samples), n_tr,
                                   random_state=seed)
    tr1 = tr1[tr_dts_idx, :]

    tr_dts_idx = CArray.randsample(CArray.arange(0, tr2.num_samples), n_tr,
                                   random_state=seed)
    tr2 = tr2[tr_dts_idx, :]

    # end train and validation dataset split

    tr1.X /= 255.0
    val.X /= 255.0
    ts.X /= 255.0
    tr2.X /= 255.0

    return tr1, val, ts, tr2
