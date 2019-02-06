from secml.utils import CUnitTest

from secml.explanation import CExplainerLocalInfluence
from secml.array import CArray
from secml.data.loader import CDataLoaderMNIST
from secml.data.splitter import CDataSplitterKFold

class CExplainerLocalInfluenceTestCases(CUnitTest):
    """Unittests interface for CExplainerLocalInfluence."""

    def _create_mnist_dataset(self, digits=[4, 9], n_tr=50, n_val=1000,
                              n_ts=1000,
                             seed=10):

        loader = CDataLoaderMNIST()

        tr = loader.load('training', digits=digits)

        ts = loader.load('testing', digits=digits, num_samples=n_ts)

        # start train and validation dataset split
        splitter = CDataSplitterKFold(num_folds=2, random_state=seed)
        splitter.compute_indices(tr)

        val_dts_idx = CArray.randsample(CArray.arange(0, tr.num_samples),
                                        n_val,
                                        random_state=seed)
        val = tr[val_dts_idx, :]

        tr_dts_idx = CArray.randsample(CArray.arange(0, tr.num_samples), n_tr,
                                       random_state=seed)
        tr = tr[tr_dts_idx, :]

        tr.X /= 255.0
        val.X /= 255.0
        ts.X /= 255.0

        return tr, val, ts

    def setUp(self):

        self._tr, self._val, self._ts = self._create_mnist_dataset()

        self._clf_creation()
        self._clf.fit(self._tr)

        clf_loss = self._clf.gradients._loss.class_type
        explanation = CExplainerLocalInfluence(self._clf, self._tr,
                                               outer_loss_idx=clf_loss)

        self.influences = explanation.explain(self._ts.X, self._ts.Y)


    def _test_explanation(self):

        self.assertEqual(self.influences.shape,
                         (self._ts.num_samples,self._tr.num_samples),
                         "The shape of the influences is wrong!")


if __name__ == '__main__':
    CUnitTest.main()
