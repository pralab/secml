from secml.testing import CUnitTest

from datetime import datetime

from secml.data.splitter import CChronologicalSplitter
from secml.data.loader import CDLRandom
from secml.data import CDataset, CDatasetHeader
from secml.array import CArray


class TestCChronologicalSplitter(CUnitTest):
    """Unit test for CChronologicalSplitter."""

    def setUp(self):

        self.ds = CDLRandom(n_samples=10, random_state=0).load()

        timestamps = CArray(['2016-02-17T10:35:58',
                             '2014-04-04T22:24:22',
                             '2016-08-07T17:10:36',
                             '2014-05-22T11:02:58',
                             '2016-07-01T07:12:34',
                             '2016-01-03T13:10:38',
                             '2014-07-28T23:42:00',
                             '2014-07-08T09:42:42',
                             '2016-05-06T18:38:08',
                             '2015-11-03T21:07:04'])

        self.ds.header = CDatasetHeader(
            timestamp=timestamps, timestamp_fmt='%Y-%m-%dT%H:%M:%S')

    def test_chronological_split(self):

        # Test splitter with default values (just seed for reproducibility)
        tts = CChronologicalSplitter(
            th_timestamp='2015',
            random_state=0)

        tr_idx, ts_idx = tts.compute_indices(self.ds)

        self.logger.info("TR IDX:\n{:}".format(tr_idx))
        self.logger.info("TS IDX:\n{:}".format(ts_idx))

        tr_idx_expected = CArray([6, 7, 3, 1])
        ts_idx_expected = CArray([9, 4, 2, 5, 0, 8])

        self.assertIsInstance(tr_idx, CArray)
        self.assertIsInstance(ts_idx, CArray)

        self.assertFalse((tr_idx != tr_idx_expected).any())
        self.assertFalse((ts_idx != ts_idx_expected).any())

        tr, ts = tts.split(self.ds)

        tr_expected = self.ds[tr_idx, :]
        ts_expected = self.ds[ts_idx, :]

        self.assertIsInstance(tr, CDataset)
        self.assertIsInstance(ts, CDataset)

        self.assert_array_equal(tr.X, tr_expected.X)
        self.assert_array_equal(tr.Y, tr_expected.Y)
        self.assert_array_equal(ts.X, ts_expected.X)
        self.assert_array_equal(ts.Y, ts_expected.Y)

        fmt = self.ds.header.timestamp_fmt
        tr_tmps = tr.header.timestamp
        ts_tmps = ts.header.timestamp

        self.assertFalse(any(map(
            lambda tstmp: datetime.strptime(tstmp, fmt) > tts.th_timestamp,
            tr_tmps)))

        self.assertFalse(any(map(
            lambda tstmp: datetime.strptime(tstmp, fmt) <= tts.th_timestamp,
            ts_tmps)))

        # Test splitter with custom integer train size
        tts = CChronologicalSplitter(
            th_timestamp='2015',
            train_size=2,
            random_state=0)

        tr_idx, ts_idx = tts.compute_indices(self.ds)

        self.logger.info("TR IDX:\n{:}".format(tr_idx))
        self.logger.info("TS IDX:\n{:}".format(ts_idx))

        tr_idx_expected = CArray([6, 7])
        ts_idx_expected = CArray([9, 4, 2, 5, 0, 8])

        self.assertFalse((tr_idx != tr_idx_expected).any())
        self.assertFalse((ts_idx != ts_idx_expected).any())

        # Test splitter with custom float train/test size
        tts = CChronologicalSplitter(
            th_timestamp='2015',
            train_size=0.25,
            test_size=0.5,
            random_state=0)

        tr_idx, ts_idx = tts.compute_indices(self.ds)

        self.logger.info("TR IDX:\n{:}".format(tr_idx))
        self.logger.info("TS IDX:\n{:}".format(ts_idx))

        tr_idx_expected = CArray([6])
        ts_idx_expected = CArray([9, 4, 2])

        self.assertFalse((tr_idx != tr_idx_expected).any())
        self.assertFalse((ts_idx != ts_idx_expected).any())

        # Test splitter with no random shuffle
        tts = CChronologicalSplitter(
            th_timestamp='2015',
            shuffle=False)

        tr_idx, ts_idx = tts.compute_indices(self.ds)

        self.logger.info("TR IDX:\n{:}".format(tr_idx))
        self.logger.info("TS IDX:\n{:}".format(ts_idx))

        tr_idx_expected = CArray([1, 3, 6, 7])
        ts_idx_expected = CArray([0, 2, 4, 5, 8, 9])

        self.assertFalse((tr_idx != tr_idx_expected).any())
        self.assertFalse((ts_idx != ts_idx_expected).any())


if __name__ == '__main__':
    CUnitTest.main()
