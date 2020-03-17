import unittest

from secml.data import CDataset
from secml.array import CArray
from secml.data.splitter import *
from secml.data.loader import CDLRandom
from secml.testing import CUnitTest


class TestCDataSplitter(CUnitTest):
    """Unit test for data splitting methods."""

    def test_kfold(self):

        ds = CDLRandom(n_samples=10, random_state=0).load()

        self.logger.info("Testing K-Fold")
        kf = CDataSplitterKFold(
            num_folds=2, random_state=5000).compute_indices(ds)

        tr_idx_expected = [CArray([1, 2, 5, 8, 9]), CArray([0, 3, 4, 6, 7])]
        ts_idx_expected = [CArray([0, 3, 4, 6, 7]), CArray([1, 2, 5, 8, 9])]

        self.assertEqual(len(kf.tr_idx), 2)
        self.assertEqual(len(kf.ts_idx), 2)

        for fold_idx in range(kf.num_folds):
            self.logger.info("{:} fold: \nTR {:} \nTS {:}"
                             "".format(fold_idx, kf.tr_idx[fold_idx],
                                       kf.ts_idx[fold_idx]))
            self.assert_array_equal(
                tr_idx_expected[fold_idx], kf.tr_idx[fold_idx])
            self.assert_array_equal(
                ts_idx_expected[fold_idx], kf.ts_idx[fold_idx])

    def test_labelkfold(self):

        ds = CDLRandom(
            n_classes=3, n_samples=10, n_informative=3, random_state=0).load()

        self.logger.info("Testing Label K-Fold")
        kf = CDataSplitterLabelKFold(num_folds=2).compute_indices(ds)

        tr_idx_expected = [CArray([1, 2, 6, 7, 8, 9]), CArray([0, 3, 4, 5])]
        ts_idx_expected = [CArray([0, 3, 4, 5]), CArray([1, 2, 6, 7, 8, 9])]

        self.assertEqual(len(kf.tr_idx), 2)
        self.assertEqual(len(kf.ts_idx), 2)

        for fold_idx in range(kf.num_folds):
            self.logger.info("{:} fold: \nTR {:} {:} \nTS {:} {:}"
                             "".format(fold_idx, kf.tr_idx[fold_idx],
                                       ds.Y[kf.tr_idx[fold_idx]],
                                       kf.ts_idx[fold_idx],
                                       ds.Y[kf.ts_idx[fold_idx]]))
            self.assert_array_equal(
                tr_idx_expected[fold_idx], kf.tr_idx[fold_idx])
            self.assert_array_equal(
                ts_idx_expected[fold_idx], kf.ts_idx[fold_idx])

    def test_openworldkfold(self):

        ds = CDLRandom(
            n_classes=3, n_samples=14, n_informative=3, random_state=0).load()

        self.logger.info("Testing Open World K-Fold")
        kf = CDataSplitterOpenWorldKFold(
            num_folds=2, n_train_samples=4,
            random_state=5000).compute_indices(ds)

        tr_idx_expected = [CArray([0, 4, 8, 12]), CArray([1, 3, 9, 13])]
        ts_idx_expected = [CArray([1, 2, 3, 5, 6, 7, 9, 10, 11, 13]),
                           CArray([0, 2, 4, 5, 6, 7, 8, 10, 11, 12])]

        self.assertEqual(len(kf.tr_idx), 2)
        self.assertEqual(len(kf.ts_idx), 2)

        self.logger.info("DS classes:\n{:}".format(ds.Y))

        for fold_idx in range(kf.num_folds):

            self.logger.info(
                "{:} fold:\nTR CLASSES {:}\nTR {:} {:}\nTS {:} {:}".format(
                    fold_idx, kf.tr_classes[fold_idx],
                    kf.tr_idx[fold_idx], ds.Y[kf.tr_idx[fold_idx]],
                    kf.ts_idx[fold_idx], ds.Y[kf.ts_idx[fold_idx]]))
            self.assert_array_equal(
                tr_idx_expected[fold_idx], kf.tr_idx[fold_idx])
            self.assert_array_equal(
                ts_idx_expected[fold_idx], kf.ts_idx[fold_idx])

    def test_openworldkfold_tr_class_skip(self):

        ds = CDataset([[1, 2], [3, 4], [5, 6],
                       [10, 20], [30, 40], [50, 60],
                       [100, 200], [300, 400], [500, 600]],
                      [1, 2, 1, 2, 2, 0, 1, 0, 2])  # class 0 has 2 samples
        # create 25 folds to increase the chance of getting the warning message
        kf = CDataSplitterOpenWorldKFold(
            num_folds=25, n_train_samples=2,
            random_state=5000).compute_indices(ds)

        self.assertEqual(len(kf.tr_idx), 25)
        self.assertEqual(len(kf.ts_idx), 25)

        for fold_tr_idx, fold_ts_idx in kf:
            self.assertTrue((ds.Y[fold_tr_idx] != 0).all())
            self.assertTrue((ds.Y[fold_ts_idx] == 0).any())

    def test_shuffle(self):

        ds = CDLRandom(n_samples=10, random_state=0).load()

        self.logger.info("Testing Shuffle ")
        kf = CDataSplitterShuffle(
            num_folds=2, train_size=0.2,
            random_state=5000).compute_indices(ds)

        tr_idx_expected = [CArray([1, 2]), CArray([9, 3])]
        ts_idx_expected = [CArray([6, 4, 7, 0, 3, 9, 5, 8]),
                           CArray([7, 5, 4, 0, 8, 2, 6, 1])]

        self.assertEqual(len(kf.tr_idx), 2)
        self.assertEqual(len(kf.ts_idx), 2)

        self.logger.info("DS classes:\n{:}".format(ds.Y))

        for fold_idx in range(kf.num_folds):
            self.logger.info("{:} fold: \nTR {:} \nTS {:}"
                             "".format(fold_idx, kf.tr_idx[fold_idx],
                                       kf.ts_idx[fold_idx]))
            self.assert_array_equal(
                tr_idx_expected[fold_idx], kf.tr_idx[fold_idx])
            self.assert_array_equal(
                ts_idx_expected[fold_idx], kf.ts_idx[fold_idx])

    def test_stratifiedkfold(self):

        ds = CDLRandom(n_samples=10, random_state=0).load()

        self.logger.info("Testing Stratified K-Fold")
        kf = CDataSplitterStratifiedKFold(
            num_folds=2, random_state=5000).compute_indices(ds)

        import sklearn
        if sklearn.__version__ < '0.22':  # TODO: REMOVE AFTER BUMPING DEPS
        # v0.22 changed the model to fix an issue related test set size
        # https://github.com/scikit-learn/scikit-learn/pull/14704
            tr_idx_expected = [CArray([4, 5, 6, 9]), CArray([0, 1, 2, 3, 7, 8])]
            ts_idx_expected = [CArray([0, 1, 2, 3, 7, 8]), CArray([4, 5, 6, 9])]
        else:
            tr_idx_expected = [CArray([1, 2, 7, 8, 9]), CArray([0, 3, 4, 5, 6])]
            ts_idx_expected = [CArray([0, 3, 4, 5, 6]), CArray([1, 2, 7, 8, 9])]

        self.assertEqual(len(kf.tr_idx), 2)
        self.assertEqual(len(kf.ts_idx), 2)

        self.logger.info("DS classes:\n{:}".format(ds.Y))

        for fold_idx in range(kf.num_folds):
            self.logger.info("{:} fold: \nTR {:} \nTS {:}"
                             "".format(fold_idx, kf.tr_idx[fold_idx],
                                       kf.ts_idx[fold_idx]))
            self.assert_array_equal(
                tr_idx_expected[fold_idx], kf.tr_idx[fold_idx])
            self.assert_array_equal(
                ts_idx_expected[fold_idx], kf.ts_idx[fold_idx])


if __name__ == '__main__':
    unittest.main()
