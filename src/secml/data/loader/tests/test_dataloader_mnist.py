from secml.testing import CUnitTest

from secml.data.loader import CDataLoaderMNIST


class TestCDataLoaderMNIST(CUnitTest):
    """Unittests for CDataLoaderMNIST."""

    def test_load(self):

        digits = (1, 5, 9)

        tr = CDataLoaderMNIST().load('training', digits=digits)

        self.logger.info(
            "Loading {:} training set samples".format(tr.num_samples))

        self.assertEqual(tr.num_samples, 18112)

        ts = CDataLoaderMNIST().load('testing', digits=digits)

        self.logger.info(
            "Loading {:} test set samples".format(ts.num_samples))

        self.assertEqual(ts.num_samples, 3036)

        n_tr = 1000
        n_ts = 1000

        tr = CDataLoaderMNIST().load(
            'training', digits=digits, num_samples=n_tr)

        self.logger.info(
            "Loading {:} training set samples".format(tr.num_samples))

        self.assertEqual(tr.num_samples, n_tr)

        ts = CDataLoaderMNIST().load(
            'testing', digits=digits, num_samples=n_ts)

        self.logger.info(
            "Loading {:} test set samples".format(ts.num_samples))

        self.assertEqual(ts.num_samples, n_ts)

        # Not enough number of samples (1666) for each desired digit
        # in the test set. ValueError should be raised
        with self.assertRaises(ValueError):
            CDataLoaderMNIST().load('testing', digits=digits, num_samples=5000)
