from secml.testing import CUnitTest

from secml.data.splitter import CTrainTestSplit
from secml.data.loader import CDLRandom
from secml.data import CDataset
from secml.array import CArray


class TestCTrainTestSplit(CUnitTest):
    """Unit test for train/test split."""

    def test_train_test_split(self):

        ds = CDLRandom(n_samples=10, random_state=0).load()

        tts = CTrainTestSplit(train_size=0.5, random_state=0, shuffle=False)

        tr_idx, ts_idx = tts.compute_indices(ds)

        self.logger.info("TR IDX:\n{:}".format(tr_idx))
        self.logger.info("TS IDX:\n{:}".format(ts_idx))

        tr_idx_expected = CArray([0, 1, 2, 3, 4])
        ts_idx_expected = CArray([5, 6, 7, 8, 9])

        self.assertIsInstance(tr_idx, CArray)
        self.assertIsInstance(ts_idx, CArray)

        self.assertFalse((tr_idx != tr_idx_expected).any())
        self.assertFalse((ts_idx != ts_idx_expected).any())

        tr, ts = tts.split(ds)

        tr_expected = ds[tr_idx, :]
        ts_expected = ds[ts_idx, :]

        self.assertIsInstance(tr, CDataset)
        self.assertIsInstance(ts, CDataset)

        self.assertFalse((tr.X != tr_expected.X).any())
        self.assertFalse((tr.Y != tr_expected.Y).any())
        self.assertFalse((ts.X != ts_expected.X).any())
        self.assertFalse((ts.Y != ts_expected.Y).any())

        self.logger.info("Testing splitting of sparse dataset")
        ds = CDLRandom(n_samples=10, random_state=0).load()

        ds = ds.tosparse()

        tts = CTrainTestSplit(train_size=0.25, random_state=0, shuffle=False)
        tr, ts = tts.split(ds)

        self.assertEqual(2, tr.num_samples)
        self.assertEqual(8, ts.num_samples)

        self.assertTrue(tr.issparse)
        self.assertTrue(ts.issparse)


if __name__ == '__main__':
    CUnitTest.main()
